module C5TB_top(

//====================HPS=================================== 
      output   [14: 0]   HPS_DDR3_ADDR,		//## HPS_DDR3 ##
      output   [ 2: 0]   HPS_DDR3_BA,
      output             HPS_DDR3_CAS_n,
      output   [ 0: 0]   HPS_DDR3_CKE,
      output             HPS_DDR3_CK_n,
      output             HPS_DDR3_CK_p,
      output   [ 0: 0]   HPS_DDR3_CS_n,
      output   [ 3: 0]   HPS_DDR3_DM,
      inout    [31: 0]   HPS_DDR3_DQ,
      inout    [ 3: 0]   HPS_DDR3_DQS_n,
      inout    [ 3: 0]   HPS_DDR3_DQS_p,
      output   [ 0: 0]   HPS_DDR3_ODT,
      output             HPS_DDR3_RAS_n,
      output             HPS_DDR3_RESET_n, 
      input              HPS_DDR3_RZQ,
      output             HPS_DDR3_WE_n,
      output             HPS_ENET_GTX_CLK,		//## HPS_ENET ##	
      inout              HPS_ENET_INT_n,		//hps_gpio_GPIO35
      output             HPS_ENET_MDC,
      inout              HPS_ENET_MDIO,
      input              HPS_ENET_RX_CLK,
      input    [ 3: 0]   HPS_ENET_RX_DATA,
      input              HPS_ENET_RX_DV,
      output   [ 3: 0]   HPS_ENET_TX_DATA,
      output             HPS_ENET_TX_EN,
      output             HPS_SDMMC_CLK,			//## HPS_SDMMC ##
      inout              HPS_SDMMC_CMD,
      inout    [ 3: 0]   HPS_SDMMC_DATA,
      input              HPS_UART_RX,			//## HPS_UART ##	
      output             HPS_UART_TX,					
      input              HPS_USB_CLKOUT,		//## HPS_USB ##
      inout    [ 7: 0]   HPS_USB_DATA,
      input              HPS_USB_DIR,
      input              HPS_USB_NXT,
      output             HPS_USB_STP,

		//////////// CLOCK //////////
      input              FPGA_CLK1_50
);

//=======================================================
//  REG/WIRE declarations
//=======================================================
wire hps_fpga_reset_n;
wire fpga_clk_50;
wire fpga_clk_cnn;
wire fpga_clk_stp;

pll_inst pll_inst0 (
		.refclk   (FPGA_CLK1_50),   		//  refclk.clk
		.rst      (~hps_fpga_reset_n),   //   reset.reset
		.outclk_0 (fpga_clk_50),
		.outclk_1 (fpga_clk_cnn),
		.outclk_2 (fpga_clk_stp)
	);

//=======================================================
//  Structural coding
//=======================================================
soc_system u0 (
	//Clock&Reset
	.clk_cnn_clk (fpga_clk_cnn),                            					//  clk_150.clk
    .clk_50_clk (fpga_clk_50),                             						//  clk.clk
	.reset_50_reset_n (hps_fpga_reset_n),                   					//  reset_150.reset_n
	.reset_cnn_reset_n (hps_fpga_reset_n),                   					//  reset_50.reset_n
	 //hps_fpga_reset_n
	.hps_0_h2f_reset_reset_n (hps_fpga_reset_n),                				//  hps_0_h2f_reset.reset_n
    //HPS ddr3
	.memory_mem_a (HPS_DDR3_ADDR),                        						//  .memory.mem_a
	.memory_mem_ba (HPS_DDR3_BA),                         						//  .mem_ba
	.memory_mem_ck (HPS_DDR3_CK_p),                       						//  .mem_ck
	.memory_mem_ck_n (HPS_DDR3_CK_n),                      						//  .mem_ck_n
	.memory_mem_cke (HPS_DDR3_CKE),                        						//  .mem_cke
	.memory_mem_cs_n (HPS_DDR3_CS_n),                      						//  .mem_cs_n
	.memory_mem_ras_n (HPS_DDR3_RAS_n),                    						//  .mem_ras_n
	.memory_mem_cas_n (HPS_DDR3_CAS_n),                    						//  .mem_cas_n
	.memory_mem_we_n (HPS_DDR3_WE_n),                      						//  .mem_we_n
	.memory_mem_reset_n (HPS_DDR3_RESET_n),                						//  .mem_reset_n
	.memory_mem_dq (HPS_DDR3_DQ),                          						//  .mem_dq
	.memory_mem_dqs (HPS_DDR3_DQS_p),                      						//  .mem_dqs
	.memory_mem_dqs_n (HPS_DDR3_DQS_n),                    						//  .mem_dqs_n
	.memory_mem_odt (HPS_DDR3_ODT),                        						//  .mem_odt
	.memory_mem_dm (HPS_DDR3_DM),                          						//  .mem_dm
	.memory_oct_rzqin (HPS_DDR3_RZQ),                         					//  .oct_rzqin
	//HPS ethernet
	.hps_0_hps_io_hps_io_emac1_inst_TX_CLK (HPS_ENET_GTX_CLK),        			//  .hps_io_emac1_inst_TX_CLK
	.hps_0_hps_io_hps_io_emac1_inst_TXD0 (HPS_ENET_TX_DATA[0]),    				//  .hps_io_emac1_inst_TXD0
	.hps_0_hps_io_hps_io_emac1_inst_TXD1 (HPS_ENET_TX_DATA[1]),    				//  .hps_io_emac1_inst_TXD1
	.hps_0_hps_io_hps_io_emac1_inst_TXD2 (HPS_ENET_TX_DATA[2]),    				//  .hps_io_emac1_inst_TXD2
	.hps_0_hps_io_hps_io_emac1_inst_TXD3 (HPS_ENET_TX_DATA[3]),    				//  .hps_io_emac1_inst_TXD3
	.hps_0_hps_io_hps_io_emac1_inst_RXD0 (HPS_ENET_RX_DATA[0]),    				//  .hps_io_emac1_inst_RXD0
	.hps_0_hps_io_hps_io_emac1_inst_MDIO (HPS_ENET_MDIO),          				//  .hps_io_emac1_inst_MDIO
	.hps_0_hps_io_hps_io_emac1_inst_MDC (HPS_ENET_MDC),         	 			//  .hps_io_emac1_inst_MDC
	.hps_0_hps_io_hps_io_emac1_inst_RX_CTL (HPS_ENET_RX_DV),          			//  .hps_io_emac1_inst_RX_CTL
	.hps_0_hps_io_hps_io_emac1_inst_TX_CTL (HPS_ENET_TX_EN),          			//  .hps_io_emac1_inst_TX_CTL
	.hps_0_hps_io_hps_io_emac1_inst_RX_CLK (HPS_ENET_RX_CLK),         			//  .hps_io_emac1_inst_RX_CLK
	.hps_0_hps_io_hps_io_emac1_inst_RXD1 (HPS_ENET_RX_DATA[1]),    				//  .hps_io_emac1_inst_RXD1
	.hps_0_hps_io_hps_io_emac1_inst_RXD2 (HPS_ENET_RX_DATA[2]),    				//  .hps_io_emac1_inst_RXD2
	.hps_0_hps_io_hps_io_emac1_inst_RXD3 (HPS_ENET_RX_DATA[3]),    				//  .hps_io_emac1_inst_RXD3
	//HPS SD card			
	.hps_0_hps_io_hps_io_sdio_inst_CMD (HPS_SDMMC_CMD),            				//	.hps_io_sdio_inst_CMD
	.hps_0_hps_io_hps_io_sdio_inst_D0 (HPS_SDMMC_DATA[0]),       				//	.hps_io_sdio_inst_D0
	.hps_0_hps_io_hps_io_sdio_inst_D1 (HPS_SDMMC_DATA[1]),       				//	.hps_io_sdio_inst_D1
	.hps_0_hps_io_hps_io_sdio_inst_CLK (HPS_SDMMC_CLK),          				//	.hps_io_sdio_inst_CLK
	.hps_0_hps_io_hps_io_sdio_inst_D2 (HPS_SDMMC_DATA[2]),       				//	.hps_io_sdio_inst_D2
	.hps_0_hps_io_hps_io_sdio_inst_D3 (HPS_SDMMC_DATA[3]),       				//	.hps_io_sdio_inst_D3
	//HPS USB	
	.hps_0_hps_io_hps_io_usb1_inst_D0 (HPS_USB_DATA[0]),       					//	.hps_io_usb1_inst_D0
	.hps_0_hps_io_hps_io_usb1_inst_D1 (HPS_USB_DATA[1]),       					//	.hps_io_usb1_inst_D1
	.hps_0_hps_io_hps_io_usb1_inst_D2 (HPS_USB_DATA[2]),       					//	.hps_io_usb1_inst_D2
	.hps_0_hps_io_hps_io_usb1_inst_D3 (HPS_USB_DATA[3]),       					//	.hps_io_usb1_inst_D3
	.hps_0_hps_io_hps_io_usb1_inst_D4 (HPS_USB_DATA[4]),       					//	.hps_io_usb1_inst_D4
	.hps_0_hps_io_hps_io_usb1_inst_D5 (HPS_USB_DATA[5]),       					//	.hps_io_usb1_inst_D5
	.hps_0_hps_io_hps_io_usb1_inst_D6 (HPS_USB_DATA[6]),       					//	.hps_io_usb1_inst_D6
	.hps_0_hps_io_hps_io_usb1_inst_D7 (HPS_USB_DATA[7]),       					//	.hps_io_usb1_inst_D7
	.hps_0_hps_io_hps_io_usb1_inst_CLK (HPS_USB_CLKOUT),        				//	.hps_io_usb1_inst_CLK
	.hps_0_hps_io_hps_io_usb1_inst_STP (HPS_USB_STP),           				//	.hps_io_usb1_inst_STP
	.hps_0_hps_io_hps_io_usb1_inst_DIR (HPS_USB_DIR),           				//	.hps_io_usb1_inst_DIR
	.hps_0_hps_io_hps_io_usb1_inst_NXT (HPS_USB_NXT),           				//	.hps_io_usb1_inst_NXT
				
	//HPS UART					
	.hps_0_hps_io_hps_io_uart0_inst_RX (HPS_UART_RX),           				//	.hps_io_uart0_inst_RX
	.hps_0_hps_io_hps_io_uart0_inst_TX (HPS_UART_TX),	           				//	.hps_io_uart0_inst_TX
	
	.hps_0_hps_io_hps_io_gpio_inst_GPIO35 (HPS_ENET_INT_n)
);


endmodule


