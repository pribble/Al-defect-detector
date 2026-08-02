// mac8x8_dsp — MAC 8×8 组合乘法单元（DSP 版）
// 独立成模块并强制 multstyle=dsp，使 Quartus 按 9×9/18×18 映射 DSP block。
// 与加法树物理隔离（模块内只有乘法、无加法），避免被融合成 mult_hlmac
// 乘加原语（每乘加 2 个 DSP block）。
// en=0 时输出 0：用于 pad 列（mac_c<0 或 >=in_w）屏蔽，禁止越界读 lb。
(* multstyle = "dsp" *) module mac8x8_dsp (
    input  wire        en,
    input  wire [7:0]  a,
    input  wire [7:0]  b,
    output wire [15:0] p
);
    assign p = (en ? $signed(a) : 8'sd0) * $signed(b);
endmodule

