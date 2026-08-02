// mac8x8_lut — MAC 8×8 组合乘法单元（LUT 版）
// 独立成模块并强制 multstyle=logic，使乘法器落在 LUT（8×8 ≈ 64 ALUT）。
// 与 DSP 版（mac8x8_dsp）配对使用：lane 0-3 走 DSP、lane 4-7 走 LUT，
// 平衡 DSP/ALM 资源（DSP 预算 32×2+24+8=96 < 112）。
(* multstyle = "logic" *) module mac8x8_lut (
    input  wire        en,
    input  wire [7:0]  a,
    input  wire [7:0]  b,
    output wire [15:0] p
);
    assign p = (en ? $signed(a) : 8'sd0) * $signed(b);
endmodule

