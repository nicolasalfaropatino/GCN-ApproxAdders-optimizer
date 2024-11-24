// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2023.2 (lin64) Build 4029153 Fri Oct 13 20:13:54 MDT 2023
// Date        : Sat Nov 23 18:57:52 2024
// Host        : nicolas-System-Product-Name running 64-bit Ubuntu 22.04.4 LTS
// Command     : write_verilog
//               /home/nicolas/Documents/GCN-ApproxAdders-optimizer/Model_deployment/static/graphs/project_1.v
// Design      : add8u_0FP
// Purpose     : This is a Verilog netlist of the current design or from a specific cell of the design. The output is an
//               IEEE 1364-2001 compliant Verilog HDL file that contains netlist information obtained from the input
//               design files.
// Device      : xc7k70tfbv676-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* STRUCTURAL_NETLIST = "yes" *)
module add8u_0FP
   (A,
    B,
    O);
  input [7:0]A;
  input [7:0]B;
  output [8:0]O;

  wire [7:0]A;
  wire [7:0]A_IBUF;
  wire [7:0]B;
  wire [7:0]B_IBUF;
  wire [8:0]O;
  wire [8:0]O_OBUF;
  wire n_120;
  wire n_132;
  wire n_78;

  IBUF \A_IBUF[0]_inst 
       (.I(A[0]),
        .O(A_IBUF[0]));
  IBUF \A_IBUF[1]_inst 
       (.I(A[1]),
        .O(A_IBUF[1]));
  IBUF \A_IBUF[2]_inst 
       (.I(A[2]),
        .O(A_IBUF[2]));
  IBUF \A_IBUF[3]_inst 
       (.I(A[3]),
        .O(A_IBUF[3]));
  IBUF \A_IBUF[4]_inst 
       (.I(A[4]),
        .O(A_IBUF[4]));
  IBUF \A_IBUF[5]_inst 
       (.I(A[5]),
        .O(A_IBUF[5]));
  IBUF \A_IBUF[6]_inst 
       (.I(A[6]),
        .O(A_IBUF[6]));
  IBUF \A_IBUF[7]_inst 
       (.I(A[7]),
        .O(A_IBUF[7]));
  IBUF \B_IBUF[0]_inst 
       (.I(B[0]),
        .O(B_IBUF[0]));
  IBUF \B_IBUF[1]_inst 
       (.I(B[1]),
        .O(B_IBUF[1]));
  IBUF \B_IBUF[2]_inst 
       (.I(B[2]),
        .O(B_IBUF[2]));
  IBUF \B_IBUF[3]_inst 
       (.I(B[3]),
        .O(B_IBUF[3]));
  IBUF \B_IBUF[4]_inst 
       (.I(B[4]),
        .O(B_IBUF[4]));
  IBUF \B_IBUF[5]_inst 
       (.I(B[5]),
        .O(B_IBUF[5]));
  IBUF \B_IBUF[6]_inst 
       (.I(B[6]),
        .O(B_IBUF[6]));
  IBUF \B_IBUF[7]_inst 
       (.I(B[7]),
        .O(B_IBUF[7]));
  OBUF \O_OBUF[0]_inst 
       (.I(O_OBUF[0]),
        .O(O[0]));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \O_OBUF[0]_inst_i_1 
       (.I0(B_IBUF[0]),
        .I1(A_IBUF[0]),
        .O(O_OBUF[0]));
  OBUF \O_OBUF[1]_inst 
       (.I(O_OBUF[1]),
        .O(O[1]));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT4 #(
    .INIT(16'h9666)) 
    \O_OBUF[1]_inst_i_1 
       (.I0(B_IBUF[1]),
        .I1(A_IBUF[1]),
        .I2(A_IBUF[0]),
        .I3(B_IBUF[0]),
        .O(O_OBUF[1]));
  OBUF \O_OBUF[2]_inst 
       (.I(O_OBUF[2]),
        .O(O[2]));
  LUT6 #(
    .INIT(64'h9996969696666666)) 
    \O_OBUF[2]_inst_i_1 
       (.I0(B_IBUF[2]),
        .I1(A_IBUF[2]),
        .I2(B_IBUF[1]),
        .I3(B_IBUF[0]),
        .I4(A_IBUF[0]),
        .I5(A_IBUF[1]),
        .O(O_OBUF[2]));
  OBUF \O_OBUF[3]_inst 
       (.I(O_OBUF[3]),
        .O(O[3]));
  LUT3 #(
    .INIT(8'h96)) 
    \O_OBUF[3]_inst_i_1 
       (.I0(A_IBUF[3]),
        .I1(B_IBUF[3]),
        .I2(n_78),
        .O(O_OBUF[3]));
  OBUF \O_OBUF[4]_inst 
       (.I(O_OBUF[4]),
        .O(O[4]));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT5 #(
    .INIT(32'hE81717E8)) 
    \O_OBUF[4]_inst_i_1 
       (.I0(n_78),
        .I1(B_IBUF[3]),
        .I2(A_IBUF[3]),
        .I3(A_IBUF[4]),
        .I4(B_IBUF[4]),
        .O(O_OBUF[4]));
  LUT6 #(
    .INIT(64'hEEE8E8E8E8888888)) 
    \O_OBUF[4]_inst_i_2 
       (.I0(B_IBUF[2]),
        .I1(A_IBUF[2]),
        .I2(A_IBUF[1]),
        .I3(A_IBUF[0]),
        .I4(B_IBUF[0]),
        .I5(B_IBUF[1]),
        .O(n_78));
  OBUF \O_OBUF[5]_inst 
       (.I(O_OBUF[5]),
        .O(O[5]));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT3 #(
    .INIT(8'h96)) 
    \O_OBUF[5]_inst_i_1 
       (.I0(A_IBUF[5]),
        .I1(B_IBUF[5]),
        .I2(n_120),
        .O(O_OBUF[5]));
  OBUF \O_OBUF[6]_inst 
       (.I(O_OBUF[6]),
        .O(O[6]));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT5 #(
    .INIT(32'h99969666)) 
    \O_OBUF[6]_inst_i_1 
       (.I0(A_IBUF[6]),
        .I1(B_IBUF[6]),
        .I2(n_120),
        .I3(B_IBUF[5]),
        .I4(A_IBUF[5]),
        .O(O_OBUF[6]));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT5 #(
    .INIT(32'hEEE8E888)) 
    \O_OBUF[6]_inst_i_2 
       (.I0(B_IBUF[4]),
        .I1(A_IBUF[4]),
        .I2(A_IBUF[3]),
        .I3(B_IBUF[3]),
        .I4(n_78),
        .O(n_120));
  OBUF \O_OBUF[7]_inst 
       (.I(O_OBUF[7]),
        .O(O[7]));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT5 #(
    .INIT(32'hE81717E8)) 
    \O_OBUF[7]_inst_i_1 
       (.I0(n_132),
        .I1(B_IBUF[6]),
        .I2(A_IBUF[6]),
        .I3(A_IBUF[7]),
        .I4(B_IBUF[7]),
        .O(O_OBUF[7]));
  OBUF \O_OBUF[8]_inst 
       (.I(O_OBUF[8]),
        .O(O[8]));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT5 #(
    .INIT(32'hFFE8E800)) 
    \O_OBUF[8]_inst_i_1 
       (.I0(A_IBUF[6]),
        .I1(B_IBUF[6]),
        .I2(n_132),
        .I3(B_IBUF[7]),
        .I4(A_IBUF[7]),
        .O(O_OBUF[8]));
  LUT3 #(
    .INIT(8'hE8)) 
    \O_OBUF[8]_inst_i_2 
       (.I0(A_IBUF[5]),
        .I1(B_IBUF[5]),
        .I2(n_120),
        .O(n_132));
endmodule
