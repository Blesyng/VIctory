//! Native Win32 startup launcher with clickable VRAM / RAM / Console buttons.
//!
//! Zero external GUI crates: uses `windows-sys` + DXGI already in the workspace,
//! working natively on Windows and under Wine 11 on Linux.

use anyhow::Result;

use crate::profile::HardwareProfile;

#[cfg(not(windows))]
pub fn show(_initial_exe: Option<std::path::PathBuf>) -> Result<Option<HardwareProfile>> {
    anyhow::bail!("interactive launcher UI is only available on Windows / Wine builds")
}

#[cfg(windows)]
pub fn show(initial_exe: Option<std::path::PathBuf>) -> Result<Option<HardwareProfile>> {
    win32::run_dialog(initial_exe)
}

#[cfg(windows)]
mod win32 {
    use std::ffi::OsString;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use std::path::PathBuf;
    use std::ptr::{null, null_mut};

    use anyhow::{Context, Result};
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
    use windows_sys::Win32::Graphics::Gdi::{
        COLOR_BTNFACE, CreateFontW, DEFAULT_CHARSET, DEFAULT_PITCH, DeleteObject, FF_SWISS,
        FW_BOLD, FW_NORMAL, GetStockObject, HBRUSH, HFONT, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        DEFAULT_QUALITY, DEFAULT_GUI_FONT,
    };
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::UI::Controls::Dialogs::{
        GetOpenFileNameW, OFN_FILEMUSTEXIST, OFN_NOCHANGEDIR, OFN_PATHMUSTEXIST, OPENFILENAMEW,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        BM_SETCHECK, BS_AUTORADIOBUTTON, BS_DEFPUSHBUTTON, BS_GROUPBOX, BS_PUSHBUTTON,
        BS_PUSHLIKE, CS_HREDRAW, CS_VREDRAW, CreateWindowExW,
        DefWindowProcW, DestroyWindow, DispatchMessageW, ES_AUTOHSCROLL, GWLP_USERDATA,
        GetMessageW, GetSystemMetrics, GetWindowLongPtrW, GetWindowTextLengthW, GetWindowTextW,
        IDC_ARROW, IsDialogMessageW, LoadCursorW, MB_ICONWARNING, MB_OK, MSG, MessageBoxW,
        PostQuitMessage, RegisterClassExW, SM_CXSCREEN, SM_CYSCREEN, SW_SHOW, SendMessageW,
        SetWindowLongPtrW, SetWindowTextW, ShowWindow, TranslateMessage, WM_COMMAND, WM_CLOSE,
        WM_DESTROY, WM_SETFONT, WNDCLASSEXW, WS_BORDER, WS_CAPTION, WS_CHILD, WS_EX_CLIENTEDGE,
        WS_GROUP, WS_MINIMIZEBOX, WS_OVERLAPPED, WS_SYSMENU, WS_TABSTOP, WS_VISIBLE,
    };

    const BST_UNCHECKED: usize = 0;
    const BST_CHECKED: usize = 1;

    use crate::profile::{HardwareProfile, RamTier, SavedConsole, VramTier, detect_tiers};

    // Control IDs
    const ID_EXE_EDIT: isize = 101;
    const ID_BROWSE: isize = 102;
    const ID_VRAM_2: isize = 201;
    const ID_VRAM_4: isize = 202;
    const ID_VRAM_8: isize = 203;
    const ID_VRAM_16: isize = 204;
    const ID_RAM_16: isize = 301;
    const ID_RAM_20: isize = 302;
    const ID_RAM_32: isize = 303;
    const ID_CON_S: isize = 401;
    const ID_CON_X: isize = 402;
    const ID_SUMMARY: isize = 501;
    const ID_DETECT: isize = 601;
    const ID_START: isize = 602;
    const ID_CANCEL: isize = 603;

    struct DetectedHardware {
        ram_bytes: u64,
        vram_bytes: u64,
        gpu_name: String,
    }

    struct DialogState {
        profile: HardwareProfile,
        detected: DetectedHardware,
        initial_dir: Option<PathBuf>,
        accepted: bool,
        hwnd_exe: HWND,
        hwnd_summary: HWND,
        vram_btns: [HWND; 4],
        ram_btns: [HWND; 3],
        con_btns: [HWND; 2],
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn detect_hardware() -> DetectedHardware {
        let mut mem = MEMORYSTATUSEX {
            dwLength: size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        // SAFETY: `mem` has `dwLength` initialized to its exact struct size.
        let ram_bytes = if unsafe { GlobalMemoryStatusEx(&mut mem) } != 0 {
            mem.ullTotalPhys
        } else {
            16 << 30
        };

        let mut best_vram = 0u64;
        let mut best_name = String::from("GPU não detectada");
        // SAFETY: standard DXGI 1.1 factory creation and adapter enumeration.
        if let Ok(factory) = unsafe { CreateDXGIFactory1::<IDXGIFactory1>() } {
            let mut i = 0u32;
            loop {
                // SAFETY: `factory` is a valid live DXGI factory.
                let Ok(adapter) = (unsafe { factory.EnumAdapters1(i) }) else {
                    break;
                };
                i += 1;
                // SAFETY: `adapter` is a valid live DXGI adapter returned above.
                if let Ok(desc) = unsafe { adapter.GetDesc1() } {
                    // Skip software adapters (DXGI_ADAPTER_FLAG_SOFTWARE = 2).
                    if (desc.Flags & 2) != 0 {
                        continue;
                    }
                    let vram = desc.DedicatedVideoMemory as u64;
                    if vram > best_vram || best_vram == 0 {
                        best_vram = vram;
                        let end = desc
                            .Description
                            .iter()
                            .position(|&c| c == 0)
                            .unwrap_or(desc.Description.len());
                        best_name = String::from_utf16_lossy(&desc.Description[..end]);
                    }
                }
            }
        }
        if best_vram == 0 {
            best_vram = 4 << 30;
        }
        DetectedHardware {
            ram_bytes,
            vram_bytes: best_vram,
            gpu_name: best_name,
        }
    }

    fn set_check(hwnd: HWND, checked: bool) {
        // SAFETY: `hwnd` is a live Win32 radio button control.
        unsafe {
            SendMessageW(
                hwnd,
                BM_SETCHECK,
                if checked { BST_CHECKED } else { BST_UNCHECKED },
                0,
            );
        }
    }

    fn set_text(hwnd: HWND, text: &str) {
        let w = wide(text);
        // SAFETY: `hwnd` is a live Win32 control and `w` is NUL-terminated.
        unsafe {
            SetWindowTextW(hwnd, w.as_ptr());
        }
    }

    fn read_text(hwnd: HWND) -> String {
        // SAFETY: `hwnd` is a live Win32 edit control.
        let len = unsafe { GetWindowTextLengthW(hwnd) };
        if len <= 0 {
            return String::new();
        }
        let mut buf = vec![0u16; len as usize + 1];
        // SAFETY: `buf` holds `len + 1` UTF-16 code units.
        let got = unsafe { GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32) };
        OsString::from_wide(&buf[..got.max(0) as usize])
            .to_string_lossy()
            .trim()
            .to_owned()
    }

    fn sync_ui(state: &DialogState) {
        set_check(state.vram_btns[0], state.profile.vram == VramTier::Gb2);
        set_check(state.vram_btns[1], state.profile.vram == VramTier::Gb4);
        set_check(state.vram_btns[2], state.profile.vram == VramTier::Gb8);
        set_check(state.vram_btns[3], state.profile.vram == VramTier::Gb16Plus);

        set_check(state.ram_btns[0], state.profile.ram == RamTier::Gb16);
        set_check(state.ram_btns[1], state.profile.ram == RamTier::Gb20);
        set_check(state.ram_btns[2], state.profile.ram == RamTier::Gb32Plus);

        set_check(
            state.con_btns[0],
            state.profile.console == SavedConsole::SeriesS,
        );
        set_check(
            state.con_btns[1],
            state.profile.console == SavedConsole::SeriesX,
        );

        set_text(state.hwnd_summary, &state.profile.summary());
    }

    fn browse_exe(owner: HWND, initial_dir: Option<&PathBuf>) -> Option<PathBuf> {
        let mut file_buf = vec![0u16; 1024];
        let filter: Vec<u16> = "Executáveis (*.exe)\0*.exe\0Todos os arquivos (*.*)\0*.*\0\0"
            .encode_utf16()
            .collect();
        let title = wide("Selecionar executável do jogo (Xbox Series)");
        let init_dir: Option<Vec<u16>> = initial_dir.map(|p| {
            p.as_os_str()
                .encode_wide()
                .chain(std::iter::once(0))
                .collect()
        });
        let mut ofn: OPENFILENAMEW = unsafe {
            // SAFETY: `OPENFILENAMEW` is a plain C struct where zero-init is valid.
            std::mem::zeroed()
        };
        ofn.lStructSize = size_of::<OPENFILENAMEW>() as u32;
        ofn.hwndOwner = owner;
        ofn.lpstrFilter = filter.as_ptr();
        ofn.lpstrFile = file_buf.as_mut_ptr();
        ofn.nMaxFile = file_buf.len() as u32;
        ofn.lpstrTitle = title.as_ptr();
        ofn.lpstrInitialDir = init_dir.as_ref().map_or(null(), |v| v.as_ptr());
        ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

        // SAFETY: all pointers inside `ofn` remain valid for the duration of the call.
        if unsafe { GetOpenFileNameW(&mut ofn) } != 0 {
            let end = file_buf
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(file_buf.len());
            if end > 0 {
                return Some(PathBuf::from(OsString::from_wide(&file_buf[..end])));
            }
        }
        None
    }

    unsafe extern "system" fn wnd_proc(
        hwnd: HWND,
        msg: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        match msg {
            WM_COMMAND => {
                let id = (wparam & 0xFFFF) as isize;
                // SAFETY: `GWLP_USERDATA` points to the live `DialogState` while the window exists.
                let ptr = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut DialogState;
                if ptr.is_null() {
                    // SAFETY: fallback to default Win32 procedure.
                    return unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) };
                }
                // SAFETY: exclusive access on the single UI thread while processing messages.
                let state = unsafe { &mut *ptr };
                match id {
                    ID_VRAM_2 => {
                        state.profile.vram = VramTier::Gb2;
                        state.profile.console = VramTier::Gb2.suggested_console();
                        sync_ui(state);
                    }
                    ID_VRAM_4 => {
                        state.profile.vram = VramTier::Gb4;
                        state.profile.console = VramTier::Gb4.suggested_console();
                        sync_ui(state);
                    }
                    ID_VRAM_8 => {
                        state.profile.vram = VramTier::Gb8;
                        state.profile.console = VramTier::Gb8.suggested_console();
                        sync_ui(state);
                    }
                    ID_VRAM_16 => {
                        state.profile.vram = VramTier::Gb16Plus;
                        state.profile.console = VramTier::Gb16Plus.suggested_console();
                        sync_ui(state);
                    }
                    ID_RAM_16 => {
                        state.profile.ram = RamTier::Gb16;
                        sync_ui(state);
                    }
                    ID_RAM_20 => {
                        state.profile.ram = RamTier::Gb20;
                        sync_ui(state);
                    }
                    ID_RAM_32 => {
                        state.profile.ram = RamTier::Gb32Plus;
                        sync_ui(state);
                    }
                    ID_CON_S => {
                        state.profile.console = SavedConsole::SeriesS;
                        sync_ui(state);
                    }
                    ID_CON_X => {
                        state.profile.console = SavedConsole::SeriesX;
                        sync_ui(state);
                    }
                    ID_DETECT => {
                        let (vram, ram) =
                            detect_tiers(state.detected.ram_bytes, state.detected.vram_bytes);
                        state.profile.vram = vram;
                        state.profile.ram = ram;
                        state.profile.console = vram.suggested_console();
                        sync_ui(state);
                    }
                    ID_BROWSE => {
                        if let Some(path) = browse_exe(hwnd, state.initial_dir.as_ref()) {
                            let s = path.to_string_lossy().into_owned();
                            set_text(state.hwnd_exe, &s);
                            state.profile.exe = Some(path);
                        }
                    }
                    ID_START => {
                        let text = read_text(state.hwnd_exe);
                        if text.is_empty() {
                            let msg_w = wide(
                                "Selecione o arquivo executável (.exe) do jogo antes de iniciar.",
                            );
                            let cap_w = wide("VIctory — Executável obrigatório");
                            // SAFETY: valid owner HWND and NUL-terminated UTF-16 strings.
                            unsafe {
                                MessageBoxW(hwnd, msg_w.as_ptr(), cap_w.as_ptr(), MB_OK | MB_ICONWARNING);
                            }
                            return 0;
                        }
                        state.profile.exe = Some(PathBuf::from(text));
                        state.accepted = true;
                        // SAFETY: closes the launcher window cleanly.
                        unsafe { DestroyWindow(hwnd) };
                    }
                    ID_CANCEL => {
                        state.accepted = false;
                        // SAFETY: closes the launcher window cleanly.
                        unsafe { DestroyWindow(hwnd) };
                    }
                    _ => {}
                }
                0
            }
            WM_CLOSE => {
                // SAFETY: destroys the window when user clicks the close button.
                unsafe { DestroyWindow(hwnd) };
                0
            }
            WM_DESTROY => {
                // SAFETY: terminates the launcher message loop.
                unsafe { PostQuitMessage(0) };
                0
            }
            // SAFETY: delegates unhandled messages to Win32 default window proc.
            _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn make_ctrl(
        class: &str,
        text: &str,
        style: u32,
        ex_style: u32,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        parent: HWND,
        id: isize,
        font: HFONT,
    ) -> HWND {
        let cls = wide(class);
        let txt = wide(text);
        // SAFETY: `parent` is a valid top-level HWND and strings are NUL-terminated.
        let ctrl = unsafe {
            CreateWindowExW(
                ex_style,
                cls.as_ptr(),
                txt.as_ptr(),
                WS_CHILD | WS_VISIBLE | style,
                x,
                y,
                w,
                h,
                parent,
                id as _,
                null_mut(),
                null(),
            )
        };
        if !ctrl.is_null() && !font.is_null() {
            // SAFETY: applies the shared UI font to the created control.
            unsafe { SendMessageW(ctrl, WM_SETFONT, font as usize, 1) };
        }
        ctrl
    }

    pub fn run_dialog(initial_exe: Option<PathBuf>) -> Result<Option<HardwareProfile>> {
        let cfg = victory_core::Config::load().ok();
        let save_path = cfg.as_ref().map(|c| c.data_dir.join("launcher.toml"));
        let detected = detect_hardware();
        let (det_vram, det_ram) = detect_tiers(detected.ram_bytes, detected.vram_bytes);

        let mut profile = save_path
            .as_deref()
            .and_then(HardwareProfile::load_saved)
            .unwrap_or_else(|| HardwareProfile {
                vram: det_vram,
                ram: det_ram,
                console: det_vram.suggested_console(),
                exe: None,
            });
        if initial_exe.is_some() {
            profile.exe = initial_exe;
        }
        let initial_dir = profile
            .exe
            .as_ref()
            .and_then(|p| p.parent().map(PathBuf::from))
            .or_else(|| cfg.as_ref().and_then(|c| c.game_dir.clone()));

        // SAFETY: retrieves the module handle of the current executable.
        let hinst = unsafe { GetModuleHandleW(null()) };
        let class_name = wide("VIctoryLauncherWindow");
        let wc = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(wnd_proc),
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: hinst,
            hIcon: null_mut(),
            // SAFETY: loads the standard Win32 arrow cursor.
            hCursor: unsafe { LoadCursorW(null_mut(), IDC_ARROW) },
            hbrBackground: (COLOR_BTNFACE + 1) as HBRUSH,
            lpszMenuName: null(),
            lpszClassName: class_name.as_ptr(),
            hIconSm: null_mut(),
        };
        // SAFETY: registers the launcher window class once.
        unsafe { RegisterClassExW(&wc) };

        let win_w = 576;
        let win_h = 505;
        // SAFETY: queries primary screen dimensions to center the window.
        let sx = unsafe { GetSystemMetrics(SM_CXSCREEN) };
        // SAFETY: queries primary screen dimensions to center the window.
        let sy = unsafe { GetSystemMetrics(SM_CYSCREEN) };
        let pos_x = ((sx - win_w) / 2).max(40);
        let pos_y = ((sy - win_h) / 2).max(40);

        let title = wide("VIctory — Configuração de Hardware (GPU / RAM)");
        // SAFETY: creates the main launcher window with a valid registered class.
        let hwnd = unsafe {
            CreateWindowExW(
                0,
                class_name.as_ptr(),
                title.as_ptr(),
                WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
                pos_x,
                pos_y,
                win_w,
                win_h,
                null_mut(),
                null_mut(),
                hinst,
                null(),
            )
        };
        anyhow::ensure!(!hwnd.is_null(), "failed to create launcher window");

        let font_face = wide("Segoe UI");
        // SAFETY: creates a standard UI font; falls back to stock GUI font on null.
        let font = unsafe {
            CreateFontW(
                -14,
                0,
                0,
                0,
                FW_NORMAL as i32,
                0,
                0,
                0,
                DEFAULT_CHARSET as u32,
                OUT_DEFAULT_PRECIS as u32,
                CLIP_DEFAULT_PRECIS as u32,
                DEFAULT_QUALITY as u32,
                (DEFAULT_PITCH | FF_SWISS) as u32,
                font_face.as_ptr(),
            )
        };
        let font_bold = unsafe {
            // SAFETY: creates a bold variant of the same UI font.
            CreateFontW(
                -15,
                0,
                0,
                0,
                FW_BOLD as i32,
                0,
                0,
                0,
                DEFAULT_CHARSET as u32,
                OUT_DEFAULT_PRECIS as u32,
                CLIP_DEFAULT_PRECIS as u32,
                DEFAULT_QUALITY as u32,
                (DEFAULT_PITCH | FF_SWISS) as u32,
                font_face.as_ptr(),
            )
        };
        let ui_font = if font.is_null() {
            // SAFETY: returns stock Win32 GUI font handle.
            unsafe { GetStockObject(DEFAULT_GUI_FONT) as HFONT }
        } else {
            font
        };
        let bold_font = if font_bold.is_null() {
            ui_font
        } else {
            font_bold
        };

        let hw_line = format!(
            "Hardware detectado: {} ({:.1} GB VRAM)  |  {:.1} GB RAM",
            detected.gpu_name,
            detected.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            detected.ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        make_ctrl("STATIC", &hw_line, 0, 0, 16, 12, 530, 22, hwnd, 0, bold_font);

        // Grupo 1: Executável
        make_ctrl(
            "BUTTON",
            "Executável do Jogo (.exe)",
            BS_GROUPBOX as u32,
            0,
            16,
            40,
            528,
            62,
            hwnd,
            0,
            ui_font,
        );
        let exe_initial = profile
            .exe
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let hwnd_exe = make_ctrl(
            "EDIT",
            &exe_initial,
            WS_BORDER | WS_TABSTOP | ES_AUTOHSCROLL as u32,
            WS_EX_CLIENTEDGE,
            28,
            64,
            406,
            26,
            hwnd,
            ID_EXE_EDIT,
            ui_font,
        );
        make_ctrl(
            "BUTTON",
            "Procurar...",
            WS_TABSTOP | BS_PUSHBUTTON as u32,
            0,
            442,
            63,
            90,
            28,
            hwnd,
            ID_BROWSE,
            ui_font,
        );

        // Grupo 2: Memória de Vídeo (VRAM)
        make_ctrl(
            "BUTTON",
            "Memória da Placa de Vídeo (GPU / VRAM)",
            BS_GROUPBOX as u32,
            0,
            16,
            110,
            528,
            66,
            hwnd,
            0,
            ui_font,
        );
        let radio_style = WS_TABSTOP | (BS_AUTORADIOBUTTON | BS_PUSHLIKE) as u32;
        let vram_btns = [
            make_ctrl(
                "BUTTON",
                "2 GB (Ultra Leve)",
                WS_GROUP | radio_style,
                0,
                28,
                134,
                120,
                30,
                hwnd,
                ID_VRAM_2,
                ui_font,
            ),
            make_ctrl(
                "BUTTON",
                "4 GB (Low-Spec)",
                radio_style,
                0,
                156,
                134,
                120,
                30,
                hwnd,
                ID_VRAM_4,
                ui_font,
            ),
            make_ctrl(
                "BUTTON",
                "8 GB (Balanceado)",
                radio_style,
                0,
                284,
                134,
                120,
                30,
                hwnd,
                ID_VRAM_8,
                ui_font,
            ),
            make_ctrl(
                "BUTTON",
                "16 GB+ (Normal)",
                radio_style,
                0,
                412,
                134,
                120,
                30,
                hwnd,
                ID_VRAM_16,
                ui_font,
            ),
        ];

        // Grupo 3: Memória RAM
        make_ctrl(
            "BUTTON",
            "Memória RAM do Computador",
            BS_GROUPBOX as u32,
            0,
            16,
            184,
            528,
            66,
            hwnd,
            0,
            ui_font,
        );
        let ram_btns = [
            make_ctrl(
                "BUTTON",
                "16 GB (Econômico)",
                WS_GROUP | radio_style,
                0,
                28,
                208,
                162,
                30,
                hwnd,
                ID_RAM_16,
                ui_font,
            ),
            make_ctrl(
                "BUTTON",
                "20 GB (Notebook)",
                radio_style,
                0,
                199,
                208,
                162,
                30,
                hwnd,
                ID_RAM_20,
                ui_font,
            ),
            make_ctrl(
                "BUTTON",
                "32 GB+ (Normal)",
                radio_style,
                0,
                370,
                208,
                162,
                30,
                hwnd,
                ID_RAM_32,
                ui_font,
            ),
        ];

        // Grupo 4: Perfil do Console
        make_ctrl(
            "BUTTON",
            "Perfil do Console Emulado",
            BS_GROUPBOX as u32,
            0,
            16,
            258,
            528,
            64,
            hwnd,
            0,
            ui_font,
        );
        let con_btns = [
            make_ctrl(
                "BUTTON",
                "Xbox Series S (Mais Leve — Recomendado 2-4 GB)",
                WS_GROUP | radio_style,
                0,
                28,
                281,
                248,
                30,
                hwnd,
                ID_CON_S,
                ui_font,
            ),
            make_ctrl(
                "BUTTON",
                "Xbox Series X (Padrão)",
                radio_style,
                0,
                284,
                281,
                248,
                30,
                hwnd,
                ID_CON_X,
                ui_font,
            ),
        ];

        // Grupo 5: Resumo ao vivo
        make_ctrl(
            "BUTTON",
            "Ajustes que serão aplicados",
            BS_GROUPBOX as u32,
            0,
            16,
            330,
            528,
            88,
            hwnd,
            0,
            ui_font,
        );
        let hwnd_summary = make_ctrl(
            "STATIC",
            "",
            0,
            0,
            28,
            352,
            504,
            58,
            hwnd,
            ID_SUMMARY,
            ui_font,
        );

        // Rodapé: ações
        make_ctrl(
            "BUTTON",
            "Auto-Detectar",
            WS_GROUP | WS_TABSTOP | BS_PUSHBUTTON as u32,
            0,
            16,
            428,
            120,
            32,
            hwnd,
            ID_DETECT,
            ui_font,
        );
        make_ctrl(
            "BUTTON",
            "Iniciar Jogo",
            WS_TABSTOP | BS_DEFPUSHBUTTON as u32,
            0,
            306,
            428,
            130,
            32,
            hwnd,
            ID_START,
            bold_font,
        );
        make_ctrl(
            "BUTTON",
            "Cancelar",
            WS_TABSTOP | BS_PUSHBUTTON as u32,
            0,
            444,
            428,
            100,
            32,
            hwnd,
            ID_CANCEL,
            ui_font,
        );

        let mut state = Box::new(DialogState {
            profile,
            detected,
            initial_dir,
            accepted: false,
            hwnd_exe,
            hwnd_summary,
            vram_btns,
            ram_btns,
            con_btns,
        });
        sync_ui(&state);

        let state_ptr: *mut DialogState = &mut *state;
        // SAFETY: `state` stays pinned on the heap for the entire message loop below.
        unsafe { SetWindowLongPtrW(hwnd, GWLP_USERDATA, state_ptr as isize) };
        // SAFETY: displays the configured launcher window.
        unsafe { ShowWindow(hwnd, SW_SHOW) };

        let mut msg: MSG = unsafe {
            // SAFETY: `MSG` is a plain C struct where zero-init is valid.
            std::mem::zeroed()
        };
        // SAFETY: standard Win32 message loop until `PostQuitMessage` on window destroy.
        while unsafe { GetMessageW(&mut msg, null_mut(), 0, 0) } > 0 {
            // SAFETY: handles Tab navigation and Enter default button inside the launcher.
            if unsafe { IsDialogMessageW(hwnd, &msg) } == 0 {
                // SAFETY: translates virtual-key messages and dispatches to `wnd_proc`.
                unsafe {
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }
        }

        if !font.is_null() {
            // SAFETY: deletes the GDI font created above after the window is destroyed.
            unsafe { DeleteObject(font) };
        }
        if !font_bold.is_null() {
            // SAFETY: deletes the bold GDI font created above after the window is destroyed.
            unsafe { DeleteObject(font_bold) };
        }

        if state.accepted {
            if let Some(path) = save_path.as_deref() {
                state.profile.save(path);
            }
            let exe = state
                .profile
                .exe
                .as_ref()
                .context("missing executable path")?;
            println!(
                "launcher       VRAM={} RAM={} console={:?} exe={}",
                state.profile.vram.label(),
                state.profile.ram.label(),
                state.profile.console,
                exe.display()
            );
            Ok(Some(state.profile))
        } else {
            Ok(None)
        }
    }
}
