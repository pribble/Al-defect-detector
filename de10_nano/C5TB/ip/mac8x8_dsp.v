// mac8x8_dsp — MAC 8×8 组合乘法单元（DSP 版）
// 独立成模块并强制 multstyle=dsp，使 Quartus 按 9×9/18×18 映射 DSP block。
// 与加法树物理隔离（模块内只有乘法、无加法），避免被融合成 mult_hlmac
// 乘加原语（每乘加 2 个 DSP block）。
// 2026-08-10：去掉 en 端口（原 en 在乘法器内部做 a 选择 mux，pad 列清零）。
// 时序：mac_c_valid_r → en → mux → 乘法 → mac_p_r 的路径穿乘法器，单拍超限
// （Quartus 按相邻沿约束，-0.167ns）。pad 列清零改为 mac_a_q 采样处门控
// （S_MAC_MUL：mac_a_q <= mac_c_valid_r ? lb_q : 0），乘法器退化为纯 a×b
// （0×b=0 等价），关键路径从"mux+乘法"缩为"纯乘法"。
(* multstyle = "dsp" *) module mac8x8_dsp (
    input  wire [7:0]  a,
    input  wire [7:0]  b,
    output wire [15:0] p
);
    assign p = $signed(a) * $signed(b);
endmodule
