# Low-Spec Memory Diet (4GB VRAM / 20GB RAM) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduzir o consumo de VRAM de 16 GB para menos de 4 GB e de RAM de 22 GB para menos de 14 GB no VIctory, viabilizando execução em laptops e hardware modesto.

**Architecture:** Aplicar corte de resolução de texturas em cascata (Global Mip-Drop para todas as texturas 2D via `over_cap`), implementar descarte LRU (Least Recently Used) de shadow buffers inativos na RAM, comprimir o orçamento de staging buffers e unificar essas otimizações em uma flag mestra `VICTORY_LOW_SPEC=1`.

**Tech Stack:** Rust 2024, Direct3D 12 (D3D12), `victory-gpu`, `victory-formats`, `windows-sys`.

---

### Task 1: Generalizar o Mip-Drop de Texturas (Corte de VRAM)

**Files:**
- Modify: `crates/victory-gpu/src/resource.rs:1145-1185`
- Test: `crates/victory-gpu/src/texture.rs`

**Context:**
Atualmente, a função `over_cap` em `crates/victory-gpu/src/resource.rs` só aplica o corte de resolução para texturas com a flag `STREAMED.get() == true` e `textures() == Textures::Stream`. Texturas criadas via `CreateCommittedResource` ou `CreatePlacedResource` diretas ainda sobem em resolução 4K/2K total, estourando a VRAM.

**Step 1: Escrever teste unitário validando o corte de mips forçado**

Criar um teste em `crates/victory-gpu/src/texture.rs` que verifica se uma textura 2D criada com dimensões grandes (ex: 2048x2048) tem seus mips superiores ignorados quando `VICTORY_TEXTURE_MAX=512`.

**Step 2: Rodar teste para verificar falha inicial**

Run: `source $HOME/.cargo/env && cd /home/anderson/trabalho/victory && cargo xwin test -p victory-gpu --lib texture::tests::test_global_mip_drop --target x86_64-pc-windows-msvc`
Expected: FAIL (pois texturas não-streamed ainda retornam skip = 0).

**Step 3: Implementar corte global em `resource.rs`**

Modificar `over_cap` em `crates/victory-gpu/src/resource.rs` para permitir que qualquer textura 2D sem flags restritivas aplique o corte de mips se `VICTORY_TEXTURE_MAX` estiver definido, independentemente de ser `STREAMED` ou colocada.

```rust
// Em crates/victory-gpu/src/resource.rs
fn over_cap(d: &ResourceDesc) -> u32 {
    static SET: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let set = *SET.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_MAX")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
    });
    // Se VICTORY_TEXTURE_MAX foi definido explicitamente pelo usuário, aplica a todas as texturas 2D!
    if set.is_none() && (!STREAMED.get() || textures() != Textures::Stream) {
        return 0;
    }
    let max = set.unwrap_or_else(|| match video_memory_percent() {
        0..70 => 1024,
        70..85 => 512,
        _ => 256,
    });
    // ... continua lógica de cálculo de skip
}
```

**Step 4: Rodar teste para validar aprovação**

Run: `source $HOME/.cargo/env && cd /home/anderson/trabalho/victory && cargo xwin test -p victory-gpu --lib --target x86_64-pc-windows-msvc`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/victory-gpu/src/resource.rs
git commit -m "feat(gpu): apply VICTORY_TEXTURE_MAX globally to all 2D textures"
```

---

### Task 2: Reduzir Staging Cache de Texturas na RAM

**Files:**
- Modify: `crates/victory-gpu/src/texture.rs:110-125`

**Context:**
O buffer de staging de texturas (`STAGING`) mantém centenas de megabytes em memória de upload do sistema. O padrão atual é 384 MiB (`VICTORY_STAGING_BUDGET_MB`), mas pode ser reduzido para 128 MiB em sistemas com pouca RAM.

**Step 1: Ajustar default e suporte a limite agressivo**

Modificar a função `budget()` em `crates/victory-gpu/src/texture.rs` para respeitar o modo low-spec: se `VICTORY_LOW_SPEC=1` estiver ativo, o padrão de staging cai para 128 MiB.

**Step 2: Validar compilação e testes**

Run: `source $HOME/.cargo/env && cd /home/anderson/trabalho/victory && cargo xwin check -p victory-gpu --target x86_64-pc-windows-msvc`
Expected: PASS com 0 erros.

**Step 3: Commit**

```bash
git add crates/victory-gpu/src/texture.rs
git commit -m "perf(gpu): lower texture staging budget for low-spec mode"
```

---

### Task 3: Descarte de Buffers Espelho Inativos (Shadow Buffer Eviction na RAM)

**Files:**
- Modify: `crates/victory-gpu/src/shadow.rs`
- Modify: `crates/victory-gpu/src/host.rs`

**Context:**
Cada buffer de geometria do jogo ganha uma cópia de host (`Shadow`). Atualmente, esses buffers nunca são liberados durante a sessão do jogo, acumulando até 10+ GB na memória RAM do sistema.

**Step 1: Adicionar rastreador de último frame de uso no `Shadow`**

No struct `Shadow`, adicionar um campo `last_used_frame: AtomicU64`. Atualizar esse contador a cada frame em que o buffer for vinculado ou desenhado.

**Step 2: Implementar rotina periódica de descarte de buffers órfãos**

Criar função `evict_idle_shadows(threshold_frames: u64)` que varre o mapa de shadows e libera buffers que não são tocados pela GPU há mais de 300 frames (~10 segundos a 30 FPS).

**Step 3: Integrar a chamada de descarte no final de cada frame**

Em `crates/victory-gpu/src/frame.rs`, disparar a limpeza a cada 60 frames.

**Step 4: Validar compilação e testes**

Run: `source $HOME/.cargo/env && cd /home/anderson/trabalho/victory && cargo xwin test -p victory-gpu --target x86_64-pc-windows-msvc`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/victory-gpu/src/shadow.rs crates/victory-gpu/src/frame.rs
git commit -m "feat(gpu): implement LRU eviction for idle shadow buffers to save RAM"
```

---

### Task 4: Flag Consolidada `VICTORY_LOW_SPEC` e Parâmetros Automáticos

**Files:**
- Modify: `apps/victory/src/run.rs`
- Modify: `/home/anderson/trabalho/run-wine-victory.sh`

**Context:**
Em vez de exigir que o usuário digite 6 variáveis de ambiente complexas a cada inicialização, criar uma flag unificada `VICTORY_LOW_SPEC=1` (ou argumento `--low-spec`) que configura automaticamente:
1. `VICTORY_TEXTURE_MAX=512` (ou `256`)
2. `VICTORY_STAGING_BUDGET_MB=128`
3. `VICTORY_AMBIENT=0.6`
4. Pula pipelines de vento/árvores distantes (`VICTORY_SKIP_PIPELINES`)
5. Desativa renderização de fios de cabelo individuais (`hair=false`)

**Step 1: Implementar expansão da flag no carregador do `run.rs`**

No `apps/victory/src/run.rs`, verificar no início de `run()` se `VICTORY_LOW_SPEC=1` está presente. Se sim, injetar as variáveis nos defaults de configuração.

**Step 2: Atualizar o script de automação `run-wine-victory.sh`**

Adicionar perfil `LOW_SPEC=1` no script utilitário local.

**Step 3: Compilar o executável final com as otimizações**

Run: `source $HOME/.cargo/env && cd /home/anderson/trabalho/victory && cargo xwin build --target x86_64-pc-windows-msvc -p victory`
Expected: Build de `victory.exe` com sucesso.

**Step 4: Testar execução do comando config sob Wine**

Run: `/home/anderson/trabalho/run-wine-victory.sh config`
Expected: Saída limpa confirmando configurações ativas.

**Step 5: Commit**

```bash
git add apps/victory/src/run.rs /home/anderson/trabalho/run-wine-victory.sh
git commit -m "feat: add unified VICTORY_LOW_SPEC profile for 4GB VRAM / 20GB RAM systems"
```
