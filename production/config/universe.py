"""
Asset Universe Configuration
============================
Defines the filtered universe of assets for trading recommendations.
"""

# S&P 500 / Large Cap universe
TRADING_UNIVERSE = [
    'AAPL','AXP','BA','CAT','CSCO','CVX','DIS','DOW','GE','GS','HD','IBM','INTC','JNJ','JPM','KO','MCD','MMM','MRK','MSFT',
    'NKE','PKE','PG','TRV','UNH','V','VZ','WMT','XOM','CSX','ETSY','A','AA','AAP','ABBV','ABT','ACM','ACN','ADM','AEP',
    'AES','AFG','AFL','AGCO','AIG','AIZ','AJG','AL','ALB','ALK','ALL','ALLY','AME','AMG','AMP','AMT','AN','ANET','AON',
    'AOS','APD','APH','ARES','ARMK','ARW','ASH','AVTR','AVY','AWI','AWK','AXTA','AYI','AZO','BAC','BAH','BALL','BAX',
    'BBY','BC','BDK','BEN','BFAM','BILL','BIO','BJ','BK','BKR','BLD','BMY','BR','BRKB','BRO','BSX','BURL','BWA','BX',
    'BYD','C','CABO','CACI','CAG','CAH','CARR','CB','CBRE','CC','CCI','CCK','CCL','CE','CF','CFG','CFR','CHD','CHE',
    'CHWY','CI','CIEN','CL','CLF','CLX','CMG','CMI','CMS','CNC','CNP','COF','COHR','COO','COP','COR','COTY','CPAY',
    'CRL','CRM','CSL','CTVA','CVNA','CVS','D','DAL','DAR','DD','DE','DECK','DELL','DG','DGX','DHI','DHR','DKS','DLB',
    'DOV','DPZ','DRI','DT','DTE','DUK','DVA','DVN','DXC','ECL','EFX','EHC','EIX','EL','ELAN','ELV','EME','EMN','EMR',
    'ENOV','EOG','EPAM','EQH','EQT','ES','ESI','ESNT','ESTC','ETN','ETR','EVR','EVRG','EW','EXC','EXP','EXPD','F',
    'FAF','FBIN','FCX','FDS','FDX','FE','FHN','FICO','FIS','FLS','FMC','FND','FNF','FTV','G','GAP','GD','GDDY','GGG',
    'GIS','GL','GLW','GM','GMED','GNRC','GPC','GPK','GPN','GT','GTLS','GWRE','GWW','H','HAL','HCA','HEI','HIG','HII',
    'HLT','HOG','HPE','HPQ','HUBB','HUBS','HUM','HUN','HWM','HXL','ICE','IEX','IFF','INGR','INSP','IP','IQV','IR',
    'IT','ITT','ITW','IVZ','J','JBL','JCI','JEF','JLL','KBR','KDP','KEY','KEYS','KKR','KMB','KMI','KMX','KNX','KR',
    'KSS','L','LAD','LDOS','LEA','LEG','LEN','LEVI','LH','LHX','LII','LLY','LMT','LNC','LOW','LPX','LUV','LVS','LW',
    'LYB','LYV','M','MA','MAN','MAS','MCK','MCO','MDT','MET','MGM','MHK','MKL','MLM','MO','MOH','MOS','MPC','MS',
    'MSCI','MSI','MSM','MTB','MTD','MTN','MTZ','NVLH','NEE','NEM','NET','NI','NOC','NOV','NOW','NRG','NSC','NUE',
    'NVST','NYT','OC','OGE','OKE','OLN','OMC','OMF','ORCL','ORI','OSK','OTIS','OXY','PANW','PAYC','PEG','PEN','PFGC',
    'PGR','PH','PHM','PII','PINS','PKG','PLD','PLNT','PNFP','PM','PNC','PCW','POST','PPG','PPL','PRU','PSA','PSTG',
    'PVH','PWR','QTWO','RCL','RF','RGA','RH','RHI','RJF','RL','RMD','RNG','ROK','ROL','ROP','RPM','RRX','RS','RSG',
    'RTX','RVTY','SAM','SCHW','SCI','SEE','SF','SHW','SITE','SJM','SLB','SMG','SNA','SNAP','SNX','SO','SPG','SPGI',
    'SER','ST','STE','STT','STZ','SWK','SYF','SYK','SYY','T','TAP','TDG','TDOC','TDY','TFC','VIRT','TFX','TGT','TC',
    'THO','TJX','TKR','TMO','TNL','TOL','TPL','TPK','TREX','TRGP','TRU','TSN','TTC','TWLO','TXT','TYL','UAL','UBER',
    'UGI','UHS','UNM','UNP','UPS','URI','USN','USFD','VAC','VEEV','VFC','VLO','VMC','VOYA','VST','VVV','VYX','W',
    'WAB','WAL','WAT','WCC','WEC','WEX','WFC','WH','WHR','WK','WM','WMB','WMS','WRB','WSM','WSO','WST','WTRG','WU',
    'XPO','XYL','YETI','YUM','ZBH','ZTS','WDAY','XEL','ZWS','XYZ','SGI','BRSL','MRSH','FSIV','AAL','ACHC','ADBE',
    'ADI','ADP','ADSK','AKAM','ALGN','ALNY','AMAT','AMD','AMGN','AMKR','AMZN','APA','APLS','ARCC','ARWR','AVGO',
    'AXON','AZTA','BIIB','BKNG','BL','BLDR','BLK','BMRN','BRKR','CACC','CAR','CASY','CDNS','CDW','CG','CGNX','CHDN',
    'CHRW','CHTR','CINF','CMCSA','CME','COLM','COST','CPRT','CROX','CRWD','CSGP','CTAS','CTSH','DBX','DDOG','DLTR',
    'DLNI','DOCU','DOX','DXCM','EA','EBAY','EEFT','ENPH','ENTG','EWBC','EXEL','EXPE','FANG','FAST','FFIV','FITB',
    'FIVE','FIVN','FLEX','FOX','FOXA','FOXF','FRPT','FSLR','FTNT','GEN','GH','GILD','GNTX','GOOG','GOOGL','HALO',
    'HAS','HBAN','HELE','HOLX','HQY','HSIC','HON','IBKR','IDXX','ILMN','INCY','INTU','IONS','IPGP','IRDM','ISRG',
    'JBHT','JKHY','KHC','KLAC','LITE','LKQ','LNT','LPLA','LRCX','LSCC','LSTR','LYFT','MANH','MAR','MASI','MAT',
    'MCHP','MDB','MDLZ','MEDP','META','MIDD','MKSI','MKTX','MNST','MPWR','MRNA','MRVL','MSTR','MU','NBIX','NDAQ',
    'NDSN','NFLX','NTAP','NTNX','NTRS','NVDA','NWL','NWSA','NXST','ODFL','OKTA','OLED','OLLI','OMCL','ON','ORLY',
    'OZK','PAYX','PCAR','PCTY','PEGA','PENN','PEP','PFG','PGNY','PODD','POOL','PTC','PYPL','QCOM','QRVO','RARE',
    'REGN','RGEN','RGLD','ROKU','ROST','RRR','SAIA','SAIC','SBUX','SLAB','SLM','SNPS','SRPT','SSNC','STLD','SWKS',
    'SYNA','TECH','TER','TMUS','TNDM','TRIP','TRMB','TROW','TSCO','TSLA','TTD','TTEK','TTWO','TXG','TXN','TXRH',
    'ULTA','UTHR','VIR','VRNS','VRSK','VRSN','VRTX','VTRS','WDC','WWD','WYNN','XRAY','ZBRA','ZD','ZG','ZION','ZM',
    'ZS','COIN',
]

# Clean up the list (remove trailing comma issues)
TRADING_UNIVERSE = [s.strip() for s in TRADING_UNIVERSE if s.strip()]
