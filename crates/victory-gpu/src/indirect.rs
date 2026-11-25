//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So active records are copied and clamped
//! on the GPU into a device-local scratch buffer, just before execution, by a
//! compute shader of ours (HLSL compiled by the system's compiler; a 5.1
//! shader with its root signature inside), leaving the title's argument and
//! count buffers untouched.
//!
//! When a count buffer is given, threads at or past `min(max, count)` return
//! immediately without charging the list's budget or reading uninitialized
//! tail slots. Each list has a budget (2^27 vertex instances or threads,
//! groups counted as 64 threads, mesh groups as 128): active records past it
//! run empty, saturating without 32-bit wraparound so a list of thousands of
//! garbage records cannot run for seconds either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint args_offset;
    uint has_count;
    uint count_offset;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint eff_count = records;
    if (has_count != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
    if (id.x >= eff_count) return;
    uint src_at = args_offset + id.x * stride;
    uint dst_at = id.x * stride;
    uint cost;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(src_args.Load(src_at), 1u << 24);
        uint instances = min(src_args.Load(src_at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw_counts = uint2(count, instances);
    } else if (kind == 2) {
        uint3 g = min(src_args.Load3(src_at), uint3(65535u, 65535u, 65535u));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64u;
        groups = g;
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
    bool over = false;
    if (cost > 0) {
        uint before;
        budget.InterlockedOr(0, 0, before);
        // Bounded retries: an unbounded loop of atomics is a GPU hang if
        // contention never settles; a record that gives up runs empty.
        bool settled = false;
        [allow_uav_condition]
        for (uint tries = 0; tries < 256 && before < BUDGET; tries++) {
            uint next = (BUDGET - before <= cost) ? BUDGET : (before + cost);
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) { settled = true; break; }
            before = prev;
        }
        over = !settled || before >= BUDGET || BUDGET - before < cost;
    }
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}
"#;

const HOIST_PASS1: &str = r#"
#define RS "RootConstants(num32BitConstants=7, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint args_offset;
    uint has_count;
    uint count_offset;
    uint slot_idx;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
// Per-clamp slot: uint4(cost, eff_count, prefix, 0) at slot_idx * 16.
RWByteAddressBuffer slots : register(u1);
static const uint BUDGET = 1u << 27;
static const uint CAP = BUDGET + 1u;
static const uint GROUPS = 1u << 20;
groupshared uint g_cost[64];
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 gtid3 : SV_GroupThreadID) {
    uint gtid = gtid3.x;
    uint eff_count = records;
    if (has_count != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
    if (id.x == 0) {
        slots.Store(slot_idx * 16 + 4, eff_count);
    }
    uint cost = 0;
    if (id.x < eff_count) {
        uint src_at = args_offset + id.x * stride;
        uint dst_at = id.x * stride;
        if (kind <= 1) {
            uint count = min(src_args.Load(src_at), 1u << 24);
            uint instances = min(src_args.Load(src_at + 4), (1u << 26) / max(count, 1u));
            cost = count * instances;
            dst_args.Store2(dst_at, uint2(count, instances));
            for (uint off = 8; off < stride; off += 4) {
                dst_args.Store(dst_at + off, src_args.Load(src_at + off));
            }
        } else if (kind == 2) {
            uint3 g = min(src_args.Load3(src_at), uint3(65535u, 65535u, 65535u));
            if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
                g = uint3(0, 0, 0);
            }
            cost = g.x * g.y * g.z * 64u;
            dst_args.Store3(dst_at, g);
            for (uint off = 12; off < stride; off += 4) {
                dst_args.Store(dst_at + off, src_args.Load(src_at + off));
            }
        } else {
            uint3 g = src_args.Load3(src_at);
            g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
            cost = g.x * g.y * g.z * 128u;
            dst_args.Store3(dst_at, g);
            for (uint off = 12; off < stride; off += 4) {
                dst_args.Store(dst_at + off, src_args.Load(src_at + off));
            }
        }
    }
    g_cost[gtid] = cost;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 32u; s > 0u; s >>= 1u) {
        if (gtid < s) {
            uint a = g_cost[gtid];
            uint b = g_cost[gtid + s];
            g_cost[gtid] = (CAP - a <= b) ? CAP : (a + b);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gtid == 0) {
        uint sum = g_cost[0];
        if (records <= 64u) {
            slots.Store(slot_idx * 16, sum);
        } else if (sum > 0u) {
            uint before;
            slots.InterlockedOr(slot_idx * 16, 0, before);
            bool settled = false;
            [allow_uav_condition]
            for (uint tries = 0; tries < 256 && before < CAP; tries++) {
                uint next = (CAP - before <= sum) ? CAP : (before + sum);
                uint prev;
                slots.InterlockedCompareExchange(slot_idx * 16, before, next, prev);
                if (prev == before) { settled = true; break; }
                before = prev;
            }
            if (!settled && before < CAP) {
                uint dummy;
                slots.InterlockedMax(slot_idx * 16, CAP, dummy);
            }
        }
    }
}
"#;

const HOIST_PREFIX: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0)"
cbuffer Info : register(b0) {
    uint num_clamps;
};
RWByteAddressBuffer slots : register(u0);
static const uint BUDGET = 1u << 27;
static const uint CAP = BUDGET + 1u;
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint running = 0;
    for (uint i = 0; i < num_clamps; i++) {
        uint c = slots.Load(i * 16);
        slots.Store(i * 16 + 8, running);
        running = (CAP - running <= c) ? CAP : (running + c);
    }
}
"#;

const HOIST_TRIM: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Info : register(b0) {
    uint num_in_chunk;
};
// uint4(slot_idx, dst_offset, kind, stride) per clamp in this chunk.
ByteAddressBuffer metas : register(t0);
RWByteAddressBuffer dst_args : register(u0);
RWByteAddressBuffer slots : register(u1);
static const uint BUDGET = 1u << 27;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid3 : SV_GroupThreadID) {
    if (gid.x >= num_in_chunk) return;
    uint tid = gtid3.x;
    uint4 m = metas.Load4(gid.x * 16);
    uint slot_idx = m.x;
    uint dst_offset = m.y;
    uint kind = m.z;
    uint stride = m.w;
    uint3 s = slots.Load3(slot_idx * 16);
    uint cost = s.x;
    uint eff_count = s.y;
    uint pref = s.z;
    if (eff_count == 0) return;
    if (pref <= BUDGET && cost <= BUDGET - pref) return;
    if (pref >= BUDGET) {
        for (uint r = tid; r < eff_count; r += 64u) {
            uint at = dst_offset + r * stride;
            if (kind <= 1) {
                dst_args.Store2(at, uint2(0, 0));
            } else {
                dst_args.Store3(at, uint3(0, 0, 0));
            }
        }
        return;
    }
    if (tid == 0) {
        uint rem = BUDGET - pref;
        for (uint r = 0; r < eff_count; r++) {
            uint at = dst_offset + r * stride;
            uint rc;
            if (kind <= 1) {
                uint2 c = dst_args.Load2(at);
                rc = c.x * c.y;
            } else if (kind == 2) {
                uint3 g = dst_args.Load3(at);
                rc = g.x * g.y * g.z * 64u;
            } else {
                uint3 g = dst_args.Load3(at);
                rc = g.x * g.y * g.z * 128u;
            }
            if (rc <= rem) {
                rem -= rc;
            } else {
                rem = 0;
                if (kind <= 1) {
                    dst_args.Store2(at, uint2(0, 0));
                } else {
                    dst_args.Store3(at, uint3(0, 0, 0));
                }
            }
        }
    }
}
"#;

struct ShaderPso {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

struct Clamp {
    single: ShaderPso,
    pass1: ShaderPso,
    prefix: ShaderPso,
    trim: ShaderPso,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    make_for(device)
}

fn compile_shader(
    device: &ID3D12Device,
    source: &str,
    name: windows::core::PCSTR,
) -> windows::core::Result<ShaderPso> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            name,
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(ShaderPso { root, pso })
    }
}

fn make_for(device: &ID3D12Device) -> windows::core::Result<Clamp> {
    Ok(Clamp {
        single: compile_shader(device, SOURCE, s!("clamp"))?,
        pass1: compile_shader(device, HOIST_PASS1, s!("clamp_pass1"))?,
        prefix: compile_shader(device, HOIST_PREFIX, s!("clamp_prefix"))?,
        trim: compile_shader(device, HOIST_TRIM, s!("clamp_trim"))?,
    })
}

/// Minimum record byte stride for a supported PC indirect argument `kind`.
fn min_stride(kind: u32) -> Option<u32> {
    match kind {
        0 => Some(16),
        1 => Some(20),
        2 | 10 => Some(12),
        _ => None,
    }
}

/// Validates record alignment and returns the scratch byte size `stride * records`.
fn validate_layout(
    (kind, stride, records): (u32, u32, u32),
    src_offset: u64,
    count_offset: Option<u64>,
) -> Option<u64> {
    let min = min_stride(kind)?;
    if stride < min || !stride.is_multiple_of(4) || !src_offset.is_multiple_of(4) {
        return None;
    }
    if count_offset.is_some_and(|off| !off.is_multiple_of(4)) {
        return None;
    }
    let bytes = u64::from(stride).checked_mul(u64::from(records))?;
    (bytes <= u64::from(u32::MAX)).then_some(bytes)
}

/// Records, in `list`, the non-destructive clamp of up to `records` records of
/// `kind` (PC argument numbering) `stride` bytes apart from `src` (and bounded
/// by the 32-bit count in `count` when given) into a device-local scratch
/// buffer, leaving that scratch buffer in `INDIRECT_ARGUMENT` at offset `0`.
/// Replaces the list's compute root signature and pipeline (callers bind
/// theirs after). `None` if the clamp or alignment is not available.
pub(crate) fn bound(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let bytes = validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    record_clamp(c, &list.list, &dst, budget, src, count, spec).then_some(dst)
}

fn record_clamp(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    budget: u64,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
) -> bool {
    let Some(bytes) = validate_layout(
        (kind, stride, records),
        src_offset,
        count.map(|(_, off)| off),
    ) else {
        return false;
    };
    // SAFETY: live D3D12 buffer resources.
    let (src_base, dst_gpu) = unsafe { (src.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
    if !src_base.is_multiple_of(4) || !dst_gpu.is_multiple_of(4) || !budget.is_multiple_of(4) {
        return false;
    }
    let (src_gpu, args_offset) = match src_offset
        .checked_add(bytes)
        .filter(|&end| end <= u64::from(u32::MAX))
    {
        Some(_) => (src_base, src_offset as u32),
        None => {
            let Some(rebased) = src_base.checked_add(src_offset) else {
                return false;
            };
            (rebased, 0)
        }
    };
    let (count_gpu, has_count, count_offset) = match count {
        Some((buf, off)) => {
            // SAFETY: a live D3D12 buffer resource.
            let base = unsafe { buf.GetGPUVirtualAddress() };
            if !base.is_multiple_of(4) {
                return false;
            }
            match off.checked_add(4).filter(|&end| end <= u64::from(u32::MAX)) {
                Some(_) => (base, 1u32, off as u32),
                None => {
                    let Some(rebased) = base.checked_add(off) else {
                        return false;
                    };
                    (rebased, 1u32, 0u32)
                }
            }
        }
        None => (src_gpu, 0u32, 0u32),
    };
    let constants = [kind, stride, records, args_offset, has_count, count_offset];
    // SAFETY: an open list; our root signature and pipeline; live source,
    // count, scratch destination, and budget buffers. `src` and `count` stay
    // in `COMMON` / `GENERIC_READ` via implicit buffer promotion while `dst`
    // transitions `COMMON -> UNORDERED_ACCESS -> INDIRECT_ARGUMENT`.
    unsafe {
        // Earlier shader writes (the title's argument and count passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                dst,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
        ]);
        if records > 0 {
            l.SetComputeRootSignature(&c.single.root);
            l.SetPipelineState(&c.single.pso);
            l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, src_gpu);
            l.SetComputeRootShaderResourceView(2, count_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            l.SetComputeRootUnorderedAccessView(4, budget);
            l.Dispatch(records.div_ceil(64), 1, 1);
        }
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                dst,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    true
}

/// How clamps of indirect draws are recorded: 0 each before its draw; 1 in
/// a list of their own before the list that draws, so that a run of draws
/// is not interleaved with our dispatches (the default); 2 as 1, with a
/// barrier for shader writes before each draw as the clamps had.
/// To find faults: 3 as 1 for mesh dispatches only, 4 as 1 for draws only,
/// 5 as 1 with a new list for the clamps always, 6 the lists cut as for 1
/// with each clamp before its draw all the same.
/// 7 as 1 with the clamps run one after another (a barrier between them).
/// 8 (default) 2-pass deterministic hoisted clamps: Pass 1 runs all clamps in
/// parallel into disjoint destination slices and per-clamp cost slots with
/// zero per-dispatch UAV barriers; `finish_hoisted` then runs one prefix sum
/// and one trim dispatch per used chunk so budget trimming is deterministic
/// without serializing the GPU across hundreds of dispatches.
/// `VICTORY_HOIST_CLAMPS=<n>` sets it; diagnostics builds read it again
/// every half second from the file `VICTORY_HOIST_FILE` names, if any.
pub(crate) fn hoist_mode() -> u8 {
    use std::sync::atomic::{AtomicU8, Ordering};
    static MODE: AtomicU8 = AtomicU8::new(u8::MAX);
    let mode = MODE.load(Ordering::Relaxed);
    if mode == u8::MAX {
        let set = std::env::var("VICTORY_HOIST_CLAMPS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        MODE.store(set, Ordering::Relaxed);
        if cfg!(feature = "diagnostics")
            && let Some(file) = std::env::var_os("VICTORY_HOIST_FILE")
        {
            let _ = std::thread::Builder::new().name("victory-hoist".into()).spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    let read = std::fs::read_to_string(&file).ok().and_then(|t| t.trim().parse::<u8>().ok());
                    let cuts = CUTS.each_ref().map(|c| c.swap(0, Ordering::Relaxed));
                    if cuts != [0, 0] {
                        tracing::info!(cuts = cuts[0], queries = cuts[1], "lists cut for hoisted clamps");
                    }
                    if let Some(n) = read.filter(|&n| n <= 8 && n != MODE.load(Ordering::Relaxed)) {
                        MODE.store(n, Ordering::Relaxed);
                        tracing::info!(mode = n, "clamps of indirect draws: mode changed");
                    }
                }
            });
        }
        return set;
    }
    mode
}

/// Cuts made for hoisted clamps, and those not made for queries that run.
pub(crate) static CUTS: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Whether clamps of indirect draws go to a list before the one that draws.
pub(crate) fn hoisting() -> bool {
    hoist_mode() != 0
}

/// As [`bound`], recorded in `clamps`, a list that runs before the one that
/// executes: the buffer and the offset of the clamped records, readable as
/// indirect arguments once `clamps` ended ([`HostList::clamp_end`]).
pub(crate) fn bound_before(
    clamps: &HostList,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
) -> Option<(ID3D12Resource, u64)> {
    let bytes = validate_layout((kind, stride, records), src_offset, count.map(|(_, off)| off))?;
    if records == 0 {
        return None;
    }
    let c = clamp()?;
    // SAFETY: live buffers.
    let (src_base, count_base) =
        unsafe { (src.GetGPUVirtualAddress(), count.map(|(b, _)| b.GetGPUVirtualAddress())) };
    // Addresses the shader's offsets (32 bits) reach, as in `record_clamp`.
    let fits = |offset: u64, len: u64| offset.checked_add(len).is_some_and(|end| end <= u64::from(u32::MAX));
    let (src_gpu, args_offset) = if fits(src_offset, bytes) {
        (src_base, src_offset as u32)
    } else {
        (src_base.checked_add(src_offset)?, 0)
    };
    let (count_gpu, has_count, count_offset) = match (count, count_base) {
        (Some((_, off)), Some(base)) if fits(off, 4) => (base, 1u32, off as u32),
        (Some((_, off)), Some(base)) => (base.checked_add(off)?, 1, 0),
        _ => (src_gpu, 0, 0),
    };
    if !src_gpu.is_multiple_of(4) || !count_gpu.is_multiple_of(4) {
        return None;
    }
    let mode = hoist_mode();
    let l = &clamps.list;
    if mode == 8 {
        if clamps.hoist_entry_count() >= crate::host::MAX_HOIST_SLOTS {
            return None;
        }
        let (_, slots_gpu) = clamps.hoist_slots_resource()?;
        if !slots_gpu.is_multiple_of(4) {
            return None;
        }
        let (chunk_idx, dst, at) = clamps.clamp_room_indexed(bytes)?;
        let slot_idx = clamps.push_hoist_entry(chunk_idx, at as u32, kind, stride)?;
        // SAFETY: a live buffer.
        let dst_gpu = unsafe { dst.GetGPUVirtualAddress() } + at;
        let constants = [kind, stride, records, args_offset, has_count, count_offset, slot_idx];
        let bind_pipeline = clamps.bind_clamp_mode(2);
        // SAFETY: an open list; our pass-1 root signature and pipeline; disjoint
        // destination slice and per-clamp slot (no cross-dispatch UAV hazard).
        unsafe {
            if bind_pipeline {
                l.SetComputeRootSignature(&c.pass1.root);
                l.SetPipelineState(&c.pass1.pso);
            }
            l.SetComputeRoot32BitConstants(0, 7, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, src_gpu);
            l.SetComputeRootShaderResourceView(2, count_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            l.SetComputeRootUnorderedAccessView(4, slots_gpu);
            l.Dispatch(records.div_ceil(64), 1, 1);
        }
        return Some((dst, at));
    }

    let (budget_res, budget) = clamps.budget_resource()?;
    if !budget.is_multiple_of(4) {
        return None;
    }
    let (dst, at) = clamps.clamp_room(bytes)?;
    // SAFETY: a live buffer.
    let dst_gpu = unsafe { dst.GetGPUVirtualAddress() } + at;
    let constants = [kind, stride, records, args_offset, has_count, count_offset];
    let bind_pipeline = clamps.bind_clamp_mode(1);
    // SAFETY: an open list; our root signature and pipeline; live source,
    // count, destination (writable: `clamp_room`) and budget buffers. The
    // clamps write apart and charge the budget atomically.
    unsafe {
        if bind_pipeline {
            l.SetComputeRootSignature(&c.single.root);
            l.SetPipelineState(&c.single.pso);
        }
        l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, src_gpu);
        l.SetComputeRootShaderResourceView(2, count_gpu);
        l.SetComputeRootUnorderedAccessView(3, dst_gpu);
        l.SetComputeRootUnorderedAccessView(4, budget);
        l.Dispatch(records.div_ceil(64), 1, 1);
        // Mode 7: order budget updates on the budget buffer specifically rather
        // than issuing a global `pResource: None` UAV barrier across all work.
        if mode == 7 {
            l.ResourceBarrier(&[uav_barrier_for(Some(budget_res))]);
        }
    }
    Some((dst, at))
}

/// Finishes any pending 2-pass hoisted clamps on `clamps` by running one
/// prefix-sum dispatch and one trim dispatch per used chunk before
/// [`HostList::clamp_end`] transitions chunks to `INDIRECT_ARGUMENT`.
pub(crate) fn finish_hoisted(clamps: &HostList) {
    let (entries, chunks) = clamps.take_hoist_batch();
    if entries.is_empty() {
        return;
    }
    let Some(c) = clamp() else { return };
    let Some((slots_res, slots_gpu)) = clamps.hoist_slots_resource() else {
        return;
    };
    let l = &clamps.list;
    let num_clamps = entries.len() as u32;

    // 1. Barrier on slots so Pass 1 cost/eff_count writes are visible to prefix sum.
    // 2. Prefix sum across all `num_clamps` entries in a single dispatch.
    // SAFETY: an open list; live `slots_res`; valid prefix root signature and PSO.
    unsafe {
        l.ResourceBarrier(&[uav_barrier_for(Some(slots_res))]);
        l.SetComputeRootSignature(&c.prefix.root);
        l.SetPipelineState(&c.prefix.pso);
        l.SetComputeRoot32BitConstants(0, 1, (&raw const num_clamps).cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, slots_gpu);
        l.Dispatch(1, 1, 1);
    }

    // 3. One batched UAV barrier for `slots_res` + every used chunk before trim.
    let mut barriers = Vec::with_capacity(chunks.len() + 1);
    barriers.push(uav_barrier_for(Some(slots_res)));
    for chunk in &chunks {
        barriers.push(uav_barrier_for(Some(chunk)));
    }
    // SAFETY: an open list; live resources; valid trim root signature and PSO.
    unsafe {
        l.ResourceBarrier(&barriers);
        l.SetComputeRootSignature(&c.trim.root);
        l.SetPipelineState(&c.trim.pso);
        l.SetComputeRootUnorderedAccessView(3, slots_gpu);
    }

    // 4. One trim dispatch per chunk holding hoisted entries (usually just 1).
    let mut chunk_metas: Vec<[u32; 4]> = Vec::new();
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        chunk_metas.clear();
        for e in &entries {
            if e.chunk == chunk_idx {
                chunk_metas.push([e.slot, e.dst_offset, e.kind, e.stride]);
            }
        }
        if chunk_metas.is_empty() {
            continue;
        }
        let meta_bytes = (chunk_metas.len() * 16) as u64;
        let Some((meta_gpu, meta_ptr)) = clamps.scratch(meta_bytes) else {
            continue;
        };
        let count = chunk_metas.len() as u32;
        // SAFETY: `scratch` returns mapped upload memory of at least `meta_bytes`,
        // valid until `clamps.reset()`; `chunk` is a live UAV buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                chunk_metas.as_ptr().cast::<u8>(),
                meta_ptr,
                meta_bytes as usize,
            );
            let chunk_gpu = chunk.GetGPUVirtualAddress();
            l.SetComputeRoot32BitConstants(0, 1, (&raw const count).cast(), 0);
            l.SetComputeRootShaderResourceView(1, meta_gpu);
            l.SetComputeRootUnorderedAccessView(2, chunk_gpu);
            l.Dispatch(count, 1, 1);
        }
    }
    // Trim replaced the list's bound compute pipeline.
    let _ = clamps.bind_clamp_mode(0);
}

/// After indirect execution (or if preparation failed after [`bound`]):
/// returns the scratch buffer to the common state for the next call.
pub(crate) fn done(list: &HostList, scratch: &ID3D12Resource) {
    done_on(&list.list, scratch);
}

fn done_on(list: &ID3D12GraphicsCommandList, scratch: &ID3D12Resource) {
    // SAFETY: an open list and a live scratch buffer in INDIRECT_ARGUMENT.
    unsafe {
        list.ResourceBarrier(&[transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    uav_barrier_for(None)
}

fn uav_barrier_for(resource: Option<&ID3D12Resource>) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: match resource {
                    // SAFETY: a borrowed pointer; the barrier never releases it
                    // (matching `crate::host::transition`).
                    Some(r) => unsafe { std::mem::transmute_copy(r) },
                    None => std::mem::ManuallyDrop::new(None),
                },
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    const WORKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    hits.InterlockedAdd(0, 1, prev);
}
"#;

    struct WarpHarness {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        clamp: Clamp,
        worker: ShaderPso,
    }

    impl WarpHarness {
        fn new() -> Option<Self> {
            // SAFETY: plain DXGI WARP query and D3D12 test harness setup.
            unsafe {
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
                let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                    return None;
                }
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                let clamp = make_for(&device).expect("clamp shader must compile on WARP");
                let worker = compile_cs(&device, WORKER_CS).expect("worker shader must compile");
                Some(Self {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    clamp,
                    worker,
                })
            }
        }

        /// CPU-mapped L0 write-back buffer matching `Shadow`'s heap type.
        fn shared_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain buffer creation and persistent CPU mapping.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
                let buffer = buffer.unwrap();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                let gpu = buffer.GetGPUVirtualAddress();
                (buffer, data.cast(), gpu)
            }
        }

        /// Device-local `DEFAULT`-heap UAV scratch buffer.
        fn default_uav(&self, size: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(64 << 10),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain device-local buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
            }
            buffer.unwrap()
        }

        fn signature(&self, kind: u32, stride: u32) -> ID3D12CommandSignature {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: valid single-argument command signature description.
            unsafe {
                self.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        }

        fn submit_and_wait(&mut self) {
            self.submitted += 1;
            // SAFETY: closing, executing, waiting on fence, and resetting our list.
            unsafe {
                self.list.Close().unwrap();
                let cmd: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }
    }

    fn compile_cs(device: &ID3D12Device, source: &str) -> windows::core::Result<ShaderPso> {
        compile_shader(device, source, s!("worker"))
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    #[test]
    fn rejects_unaligned_or_undersized_layouts() {
        assert_eq!(validate_layout((0, 16, 4), 0, None), Some(64));
        assert_eq!(validate_layout((1, 20, 4), 64, Some(128)), Some(80));
        assert_eq!(validate_layout((2, 12, 0), 0, Some(0)), Some(0));
        assert_eq!(validate_layout((10, 16, 2), 8, None), Some(32));
        // Undersized strides for each kind.
        assert_eq!(validate_layout((0, 12, 1), 0, None), None);
        assert_eq!(validate_layout((1, 16, 1), 0, None), None);
        assert_eq!(validate_layout((2, 8, 1), 0, None), None);
        assert_eq!(validate_layout((10, 8, 1), 0, None), None);
        // Unknown kind.
        assert_eq!(validate_layout((3, 16, 1), 0, None), None);
        // Unaligned stride, source offset, or count offset.
        assert_eq!(validate_layout((0, 18, 1), 0, None), None);
        assert_eq!(validate_layout((0, 16, 1), 2, None), None);
        assert_eq!(validate_layout((0, 16, 1), 0, Some(6)), None);
        // Total byte size exceeding 32-bit ByteAddressBuffer indexing.
        assert_eq!(validate_layout((1, 20, u32::MAX), 0, None), None);
    }

    /// WARP D3D12 execution test proving:
    /// (a) `countBuffer` with a small count (`3`) and `4,093` garbage tail slots
    ///     (`max = 4096`) does NOT blow `BUDGET` or zero valid dispatches in
    ///     this or subsequent `ExecuteIndirect` calls in the same list,
    /// (b) `src_args` (and `count_buf`) are NOT mutated in-place, and
    /// (c) full `stride` bytes (`stride = 24` > `12`) are copied for active
    ///     records into `dst_args` at offset `0`.
    #[test]
    fn warp_count_buffer_ignores_garbage_tail_and_preserves_source() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 24; // 6 dwords: 3 dispatch group counts + 3 extra payload dwords
        const MAX: u32 = 4096;
        const ARGS_OFFSET: usize = 64; // non-zero offset with sentinel prefix
        const COUNT_OFFSET_1: usize = 128;
        const COUNT_OFFSET_2: usize = 192;

        let src_bytes = ARGS_OFFSET + (STRIDE as usize) * (MAX as usize) + 64;
        let (src_buf, src_ptr, _) = warp.shared_buffer(src_bytes as u64);
        let (count_buf, count_ptr, _) = warp.shared_buffer(512);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer((STRIDE * 8) as u64);
        let dst_buf = warp.default_uav(u64::from(STRIDE) * u64::from(MAX));
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        // Fill the entire source buffer with non-zero sentinel bytes, then
        // populate 3 valid records at `ARGS_OFFSET` and 4,093 huge garbage
        // records in `3..MAX` that would exhaust `BUDGET` ~2,000x over if read.
        let mut initial_src = vec![0xA5u8; src_bytes];
        let write_record = |buf: &mut [u8], idx: usize, words: [u32; 6]| {
            let base = ARGS_OFFSET + idx * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                buf[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        };
        write_record(
            &mut initial_src,
            0,
            [3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666],
        );
        write_record(
            &mut initial_src,
            1,
            [5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC],
        );
        // Record 2 exceeds 65,535 in X -> clamped to (65535, 1, 1).
        write_record(
            &mut initial_src,
            2,
            [100_000, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567],
        );
        for slot in 3..MAX as usize {
            // Each garbage slot would clamp to 65535 * 16 * 1 = 1,048,560 groups
            // (`cost = 67,107,840 ~= BUDGET / 2`), exhausting `BUDGET` immediately!
            write_record(
                &mut initial_src,
                slot,
                [0xFFFF_FFFF, 16, 1, 0xDEAD_BEEF, 0xBAD0_C0DE, slot as u32],
            );
        }

        // Second indirect call's source records (at offset 0 of a second buffer,
        // or later in `count_buf`): put 2 active records + garbage tail in a
        // second range to run back-to-back in the SAME command list.
        let mut initial_count = vec![0x5Au8; 512];
        initial_count[COUNT_OFFSET_1..COUNT_OFFSET_1 + 4].copy_from_slice(&3u32.to_le_bytes());
        initial_count[COUNT_OFFSET_2..COUNT_OFFSET_2 + 4].copy_from_slice(&2u32.to_le_bytes());

        // SAFETY: buffers are mapped with exact capacities above.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, src_bytes);
            std::ptr::copy_nonoverlapping(initial_count.as_ptr(), count_ptr, 512);
            (budget_ptr as *mut u32).write_volatile(0);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (STRIDE * 8) as usize);
        }

        // Record Call #1 (`eff_count = 3` out of `MAX = 4096`), copy the first
        // 5 records of `dst_buf` out for inspection, then record Call #2
        // (`eff_count = 2` out of `MAX = 4096`) reusing `dst_buf` in the same list!
        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_1 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_1 as u64,
            );
        }
        done_on(&warp.list, &dst_buf);

        // Snapshot `dst_buf` after Call #1 (from `COMMON`, copy 5 * STRIDE bytes,
        // then return `dst_buf` to `COMMON` before Call #2).
        // SAFETY: open command list; `dst_buf` returned to `COMMON` by `done_on`.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, u64::from(STRIDE * 5));
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }

        // Call #2 in the SAME command list (`eff_count = 2` -> records 0 and 1:
        // 6 + 5 = 11 groups). Would run empty if Call #1's tail blew `BUDGET`!
        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_2 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_2 as u64,
            );
        }
        done_on(&warp.list, &dst_buf);

        warp.submit_and_wait();

        // SAFETY: GPU execution finished; read mapped host results.
        let (hits, spent, after_src, after_count, clamped_words) = unsafe {
            let hits = (hits_ptr as *const u32).read_volatile();
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), src_bytes).to_vec();
            let after_count = std::slice::from_raw_parts(count_ptr.cast_const(), 512).to_vec();
            let words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                (STRIDE as usize / 4) * 5,
            )
            .to_vec();
            (hits, spent, after_src, after_count, words)
        };

        // Call #1: (3*2*1) + (5*1*1) + (65535*1*1) = 65,546 groups.
        // Call #2: (3*2*1) + (5*1*1) = 11 groups.
        let expected_groups = (6 + 5 + 65_535) + (6 + 5);
        assert_eq!(
            hits, expected_groups,
            "valid indirect dispatches must all execute"
        );
        assert_eq!(
            spent,
            expected_groups * 64,
            "budget must reflect only active records below countBuffer"
        );
        assert_eq!(
            after_src, initial_src,
            "src_args must never be mutated in-place"
        );
        assert_eq!(
            after_count, initial_count,
            "count_buf must never be mutated"
        );

        // Check `dst_args` records 0..3: clamped group counts + preserved extra stride words.
        assert_eq!(
            &clamped_words[0..6],
            &[3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666]
        );
        assert_eq!(
            &clamped_words[6..12],
            &[5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC]
        );
        assert_eq!(
            &clamped_words[12..18],
            &[65_535, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567]
        );
        // Inactive slot 3 was never copied from `src_args`.
        assert_ne!(clamped_words[21], 0xDEAD_BEEF);
    }

    /// WARP D3D12 test proving `BUDGET` saturates without 32-bit unsigned
    /// wraparound across >64 huge indexed-draw records (`count = None`), while
    /// preserving `StartIndexLocation`, `BaseVertexLocation`,
    /// `StartInstanceLocation`, and custom stride padding, and leaving
    /// `src_args` untouched.
    #[test]
    fn warp_budget_saturates_without_wraparound_and_preserves_draw_fields() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_INDEXED: u32 = 1;
        const STRIDE: u32 = 24; // 5 indexed-draw words + 1 trailing custom word
        const RECORDS: u32 = 128; // 2 waves of 64: old InterlockedAdd wrapped 2^32 at 64!
        const BUDGET_CAP: u32 = 1 << 27;

        let total_bytes = (STRIDE * RECORDS) as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let dst_buf = warp.default_uav(total_bytes as u64);

        let mut initial_src = vec![0u8; total_bytes];
        for i in 0..RECORDS as usize {
            let words: [u32; 6] = [
                u32::MAX,                    // IndexCount -> clamps to 1 << 24
                u32::MAX,                    // InstanceCount -> clamps to 4 (cost = 1 << 26)
                1000 + i as u32,             // StartIndexLocation
                (-200i32 + i as i32) as u32, // BaseVertexLocation
                5000 + i as u32,             // StartInstanceLocation
                0xCAFE_0000 | (i as u32),    // Extra stride word
            ];
            let base = i * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                initial_src[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }

        // SAFETY: mapped buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, total_bytes);
            (budget_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, total_bytes);
        }

        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_INDEXED, STRIDE, RECORDS),
        ));
        done_on(&warp.list, &dst_buf);

        // SAFETY: open command list; copy clamped scratch buffer for verification.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total_bytes as u64);
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        warp.submit_and_wait();

        // SAFETY: GPU work completed.
        let (spent, after_src, out_words) = unsafe {
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), total_bytes).to_vec();
            let out_words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                total_bytes / 4,
            )
            .to_vec();
            (spent, after_src, out_words)
        };

        assert_eq!(
            spent, BUDGET_CAP,
            "budget must saturate at BUDGET without wrapping"
        );
        assert_eq!(after_src, initial_src, "src_args must remain untouched");

        let mut kept = 0usize;
        let mut zeroed = 0usize;
        for i in 0..RECORDS as usize {
            let rec = &out_words[i * 6..(i + 1) * 6];
            if rec[0] == (1 << 24) && rec[1] == 4 {
                kept += 1;
            } else {
                assert_eq!((rec[0], rec[1]), (0, 0), "excess draw must be zeroed");
                zeroed += 1;
            }
            assert_eq!(rec[2], 1000 + i as u32);
            assert_eq!(rec[3], (-200i32 + i as i32) as u32);
            assert_eq!(rec[4], 5000 + i as u32);
            assert_eq!(rec[5], 0xCAFE_0000 | (i as u32));
        }
        assert_eq!((kept, zeroed), (2, (RECORDS as usize) - 2));
    }

    /// Exercises `HostList::indirect_scratch`, `bound`, `done`, `Host::execute`,
    /// and `HostList::reset` end-to-end on the host device.
    #[test]
    fn host_list_reuses_and_grows_scratch_across_submissions() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let Some((src_buf, src_ptr)) = host.shared_buffer(256 << 10) else {
            return;
        };
        let Some((count_buf, count_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped host buffers.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0x7F, 256 << 10);
            (count_ptr as *mut u32).write_volatile(2);
        }

        // First two calls fit inside the initial 64 KiB scratch buffer and
        // must reuse the exact same `ID3D12Resource`.
        let b1 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (0, 16, 64)).unwrap();
        done(&list, &b1);
        let b2 = bound(&list, (&src_buf, 64), Some((&count_buf, 0)), (1, 20, 128)).unwrap();
        done(&list, &b2);
        assert_eq!(
            b1.as_raw(),
            b2.as_raw(),
            "scratch buffer within capacity must be reused"
        );

        // A call needing >64 KiB grows the scratch buffer mid-list while keeping
        // `b1` alive until reset.
        let b3 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 5000)).unwrap();
        done(&list, &b3);
        assert_ne!(
            b3.as_raw(),
            b1.as_raw(),
            "larger request must grow scratch buffer"
        );

        list.close();
        host.execute(&[&list]);
        list.reset();

        // After reset, requests up to the grown high-water mark reuse `b3` with
        // zero new allocations.
        let b4 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 4096)).unwrap();
        done(&list, &b4);
        assert_eq!(
            b4.as_raw(),
            b3.as_raw(),
            "grown scratch buffer must persist across reset"
        );
        list.close();
        host.execute(&[&list]);
        list.reset();
    }

    /// WARP D3D12 test proving the 2-pass hoisted clamp (mode 8) runs multiple
    /// Pass-1 dispatches with ZERO UAV barriers between them, computes exact
    /// prefix sums in `HOIST_PREFIX`, and deterministically preserves earlier
    /// clamps, trims the single boundary clamp in exact record order `0..n`,
    /// and zeroes later clamps in `HOIST_TRIM`.
    #[test]
    fn warp_two_pass_hoisted_clamps_are_deterministic_without_per_dispatch_barriers() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_INDEXED: u32 = 1;
        const STRIDE: u32 = 20;
        const BUDGET_CAP: u32 = 1 << 27; // 134,217,728

        // 4 hoisted clamps with costs:
        // - Clamp 0: 1 record  of 16M * 3 = 48M  (prefix 0   -> KEEP)
        // - Clamp 1: 1 record  of 16M * 3 = 48M  (prefix 48M -> KEEP; remaining = 38M)
        // - Clamp 2: 4 records of 16M * 1 = 16M each (64M total, prefix 96M -> BOUNDARY:
        //            records 0 & 1 = 32M <= 38M KEEP; records 2 & 3 ZEROED!)
        // - Clamp 3: 2 records of 16M * 1 = 16M each (prefix 160M -> ZERO ALL)
        let record_cost_16m: u32 = 1 << 24;
        let clamp_specs: [(u32, &[(u32, u32)]); 4] = [
            (0, &[(record_cost_16m, 3)]),
            (64, &[(record_cost_16m, 3)]),
            (
                128,
                &[
                    (record_cost_16m, 1),
                    (record_cost_16m, 1),
                    (record_cost_16m, 1),
                    (record_cost_16m, 1),
                ],
            ),
            (256, &[(record_cost_16m, 1), (record_cost_16m, 1)]),
        ];

        let total_bytes = 512usize;
        let (src_buf, src_ptr, src_gpu) = warp.shared_buffer(total_bytes as u64);
        let (slots_buf, slots_ptr, slots_gpu) = warp.shared_buffer(256);
        let (_meta_buf, meta_ptr, meta_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let dst_buf = warp.default_uav(total_bytes as u64);

        let mut initial_src = vec![0u8; total_bytes];
        for (slot, (offset, recs)) in clamp_specs.iter().enumerate() {
            for (r, &(cnt, inst)) in recs.iter().enumerate() {
                let base = (*offset as usize) + r * (STRIDE as usize);
                let words = [cnt, inst, 100 + slot as u32, r as u32, 0xABCD_0000 | r as u32];
                for (w, v) in words.into_iter().enumerate() {
                    initial_src[base + w * 4..base + (w + 1) * 4]
                        .copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        let mut metas = [[0u32; 4]; 4];
        for (slot, (offset, _)) in clamp_specs.iter().enumerate() {
            metas[slot] = [slot as u32, *offset, KIND_INDEXED, STRIDE];
        }

        // SAFETY: mapped host buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, total_bytes);
            std::ptr::write_bytes(slots_ptr, 0, 256);
            std::ptr::copy_nonoverlapping(metas.as_ptr().cast::<u8>(), meta_ptr, 64);
            std::ptr::write_bytes(readback_ptr, 0, total_bytes);
        }

        // SAFETY: open command list, live resources, valid compiled clamp shaders.
        unsafe {
            let dst_gpu_base = dst_buf.GetGPUVirtualAddress();
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            // Pass 1: all 4 dispatches back-to-back with ZERO barriers between them!
            warp.list.SetComputeRootSignature(&warp.clamp.pass1.root);
            warp.list.SetPipelineState(&warp.clamp.pass1.pso);
            for (slot, (offset, recs)) in clamp_specs.iter().enumerate() {
                let n = recs.len() as u32;
                let constants = [KIND_INDEXED, STRIDE, n, *offset, 0u32, 0u32, slot as u32];
                warp.list
                    .SetComputeRoot32BitConstants(0, 7, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, src_gpu);
                warp.list.SetComputeRootShaderResourceView(2, src_gpu);
                warp.list
                    .SetComputeRootUnorderedAccessView(3, dst_gpu_base + u64::from(*offset));
                warp.list.SetComputeRootUnorderedAccessView(4, slots_gpu);
                warp.list.Dispatch(n.div_ceil(64), 1, 1);
            }
            // Prefix sum (1 barrier + 1 dispatch).
            let num_clamps = 4u32;
            warp.list
                .ResourceBarrier(&[uav_barrier_for(Some(&slots_buf))]);
            warp.list.SetComputeRootSignature(&warp.clamp.prefix.root);
            warp.list.SetPipelineState(&warp.clamp.prefix.pso);
            warp.list
                .SetComputeRoot32BitConstants(0, 1, (&raw const num_clamps).cast(), 0);
            warp.list.SetComputeRootUnorderedAccessView(1, slots_gpu);
            warp.list.Dispatch(1, 1, 1);

            // Trim (1 batched barrier + 1 dispatch for the whole chunk).
            warp.list.ResourceBarrier(&[
                uav_barrier_for(Some(&slots_buf)),
                uav_barrier_for(Some(&dst_buf)),
            ]);
            warp.list.SetComputeRootSignature(&warp.clamp.trim.root);
            warp.list.SetPipelineState(&warp.clamp.trim.pso);
            warp.list
                .SetComputeRoot32BitConstants(0, 1, (&raw const num_clamps).cast(), 0);
            warp.list.SetComputeRootShaderResourceView(1, meta_gpu);
            warp.list
                .SetComputeRootUnorderedAccessView(2, dst_gpu_base);
            warp.list.SetComputeRootUnorderedAccessView(3, slots_gpu);
            warp.list.Dispatch(num_clamps, 1, 1);

            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total_bytes as u64);
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        let _ = src_buf;
        warp.submit_and_wait();

        // SAFETY: GPU execution finished; read mapped results.
        let (slot_words, out_bytes, after_src) = unsafe {
            let sw = std::slice::from_raw_parts(slots_ptr.cast::<u32>().cast_const(), 16).to_vec();
            let out = std::slice::from_raw_parts(readback_ptr.cast_const(), total_bytes).to_vec();
            let src = std::slice::from_raw_parts(src_ptr.cast_const(), total_bytes).to_vec();
            (sw, out, src)
        };
        assert_eq!(after_src, initial_src, "src_args must remain untouched");

        // Verify per-clamp [cost, eff_count, prefix] computed by Pass 1 + Prefix:
        assert_eq!(&slot_words[0..3], &[3 << 24, 1, 0]);
        assert_eq!(&slot_words[4..7], &[3 << 24, 1, 3 << 24]);
        assert_eq!(&slot_words[8..11], &[4 << 24, 4, 6 << 24]);
        assert_eq!(&slot_words[12..15], &[2 << 24, 2, BUDGET_CAP + 1]);

        let read_rec = |offset: usize, r: usize| -> [u32; 5] {
            let base = offset + r * (STRIDE as usize);
            std::array::from_fn(|w| {
                u32::from_le_bytes(out_bytes[base + w * 4..base + (w + 1) * 4].try_into().unwrap())
            })
        };
        // Clamp 0 & 1: fully kept.
        assert_eq!(read_rec(0, 0), [record_cost_16m, 3, 100, 0, 0xABCD_0000]);
        assert_eq!(read_rec(64, 0), [record_cost_16m, 3, 101, 0, 0xABCD_0000]);
        // Clamp 2 (boundary): records 0 & 1 kept, records 2 & 3 deterministically zeroed.
        assert_eq!(read_rec(128, 0), [record_cost_16m, 1, 102, 0, 0xABCD_0000]);
        assert_eq!(read_rec(128, 1), [record_cost_16m, 1, 102, 1, 0xABCD_0001]);
        assert_eq!(read_rec(128, 2), [0, 0, 102, 2, 0xABCD_0002]);
        assert_eq!(read_rec(128, 3), [0, 0, 102, 3, 0xABCD_0003]);
        // Clamp 3 (past budget): all records zeroed.
        assert_eq!(read_rec(256, 0), [0, 0, 103, 0, 0xABCD_0000]);
        assert_eq!(read_rec(256, 1), [0, 0, 103, 1, 0xABCD_0001]);
    }
}
