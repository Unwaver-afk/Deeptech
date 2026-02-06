{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 HelveticaNeue-Bold;\f1\fnil\fcharset0 HelveticaNeue;\f2\fnil\fcharset0 Menlo-Regular;
\f3\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;\red154\green154\blue154;}
{\*\expandedcolortbl;;\csgray\c66667;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid1\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab560
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0  Dataset Overview\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 This project utilizes the 
\f0\b WM-811K Wafer Map Dataset
\f1\b0  (Mixed-Type Defect Variant), a standard benchmark for semiconductor fault detection.\
\pard\pardeftab560\pardirnatural\partightenfactor0
\ls1\ilvl0
\f2\fs18 \cf0 {\listtext	\uc0\u8226 	}
\f0\b\fs26 Source:
\f1\b0  Real-world wafer maps collected from wafer fabrication.\
\ls1\ilvl0
\f2\fs18 {\listtext	\uc0\u8226 	}
\f0\b\fs26 Format:
\f1\b0  Compressed NumPy archive (.npz) for efficient storage.\
\ls1\ilvl0
\f2\fs18 {\listtext	\uc0\u8226 	}
\f0\b\fs26 Volume:
\f1\b0  ~38,000 Wafer Maps.\
\ls1\ilvl0
\f2\fs18 {\listtext	\uc0\u8226 	}
\f0\b\fs26 Resolution:
\f1\b0  $52 \\times 52$ grid per wafer (Upscaled to $224 \\times 224$ for Vision Transformer).\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0  Feature Engineering (The "RGB" Trick)\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Raw wafer maps are sparse matrices containing discrete values: \{0: Background, 1: Normal Die, 2: Defect\}.\
To leverage the power of 
\f0\b Vision Transformers (DinoV2)
\f1\b0 , we apply a domain-specific 
\f0\b Categorical-to-RGB Mapping
\f1\b0  that converts abstract data into high-contrast visual features:\

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trbrdrt\brdrnil \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalt \clshdrawnil \clbrdrt\brdrs\brdrw20\brdrcf2 \clbrdrl\brdrs\brdrw20\brdrcf2 \clbrdrb\brdrs\brdrw20\brdrcf2 \clbrdrr\brdrs\brdrw20\brdrcf2 \clpadt20 \clpadl100 \clpadb20 \clpadr100 \gaph\cellx2880
\clvertalt \clshdrawnil \clbrdrt\brdrs\brdrw20\brdrcf2 \clbrdrl\brdrs\brdrw20\brdrcf2 \clbrdrb\brdrs\brdrw20\brdrcf2 \clbrdrr\brdrs\brdrw20\brdrcf2 \clpadt20 \clpadl100 \clpadb20 \clpadr100 \gaph\cellx5760
\clvertalt \clshdrawnil \clbrdrt\brdrs\brdrw20\brdrcf2 \clbrdrl\brdrs\brdrw20\brdrcf2 \clbrdrb\brdrs\brdrw20\brdrcf2 \clbrdrr\brdrs\brdrw20\brdrcf2 \clpadt20 \clpadl100 \clpadb20 \clpadr100 \gaph\cellx8640
\pard\intbl\itap1\pardeftab560\slleading20\partightenfactor0

\f0\b \cf0 Raw Value
\f3\b0\fs24 \cell 
\pard\intbl\itap1\pardeftab560\slleading20\partightenfactor0

\f0\b\fs26 \cf0 Meaning
\f3\b0\fs24 \cell 
\pard\intbl\itap1\pardeftab560\slleading20\partightenfactor0

\f0\b\fs26 \cf0 Processed Color (RGB)
\f3\b0\fs24 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalt \clshdrawnil \clbrdrt\brdrs\brdrw20\brdrcf2 \clbrdrl\brdrs\brdrw20\brdrcf2 \clbrdrb\brdrs\brdrw20\brdrcf2 \clbrdrr\brdrs\brdrw20\brdrcf2 \clpadt20 \clpadl100 \clpadb20 \clpadr100 \gaph\cellx2880
\clvertalt \clshdrawnil \clbrdrt\brdrs\brdrw20\brdrcf2 \clbrdrl\brdrs\brdrw20\brdrcf2 \clbrdrb\brdrs\brdrw20\brdrcf2 \clbrdrr\brdrs\brdrw20\brdrcf2 \clpadt20 \clpadl100 \clpadb20 \clpadr100 \gaph\cellx5760
\clvertalt \clshdrawnil \clbrdrt\brdrs\brdrw20\brdrcf2 \clbrdrl\brdrs\brdrw20\brdrcf2 \clbrdrb\brdrs\brdrw20\brdrcf2 \clbrdrr\brdrs\brdrw20\brdrcf2 \clpadt20 \clpadl100 \clpadb20 \clpadr100 \gaph\cellx8640
\pard\intbl\itap1\pardeftab560\slleading20\partightenfactor0

\f0\b\fs26 \cf0 0
\f3\b0\fs24 \cell 
\pard\intbl\itap1\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Background
\f3\fs24 \cell 
\pard\intbl\itap1\pardeftab560\slleading20\partightenfactor0

\f0\b\fs26 \cf0 White
\f1\b0  (Empty Space)
\f3\fs24 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trbrdrl\brdrnil \trbrdrr\brdrnil 
\clvertalt \clshdrawnil \clbrdrt\brdrs\brdrw20\brdrcf2 \clbrdrl\brdrs\brdrw20\brdrcf2 \clbrdrb\brdrs\brdrw20\brdrcf2 \clbrdrr\brdrs\brdrw20\brdrcf2 \clpadt20 \clpadl100 \clpadb20 \clpadr100 \gaph\cellx2880
\clvertalt \clshdrawnil \clbrdrt\brdrs\brdrw20\brdrcf2 \clbrdrl\brdrs\brdrw20\brdrcf2 \clbrdrb\brdrs\brdrw20\brdrcf2 \clbrdrr\brdrs\brdrw20\brdrcf2 \clpadt20 \clpadl100 \clpadb20 \clpadr100 \gaph\cellx5760
\clvertalt \clshdrawnil \clbrdrt\brdrs\brdrw20\brdrcf2 \clbrdrl\brdrs\brdrw20\brdrcf2 \clbrdrb\brdrs\brdrw20\brdrcf2 \clbrdrr\brdrs\brdrw20\brdrcf2 \clpadt20 \clpadl100 \clpadb20 \clpadr100 \gaph\cellx8640
\pard\intbl\itap1\pardeftab560\slleading20\partightenfactor0

\f0\b\fs26 \cf0 1
\f3\b0\fs24 \cell 
\pard\intbl\itap1\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Normal Die
\f3\fs24 \cell 
\pard\intbl\itap1\pardeftab560\slleading20\partightenfactor0

\f0\b\fs26 \cf0 Soft Teal
\f1\b0  (Silicon Structure)
\f3\fs24 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trbrdrl\brdrnil \trbrdrt\brdrnil \trbrdrr\brdrnil 
\clvertalt \clshdrawnil \clbrdrt\brdrs\brdrw20\brdrcf2 \clbrdrl\brdrs\brdrw20\brdrcf2 \clbrdrb\brdrs\brdrw20\brdrcf2 \clbrdrr\brdrs\brdrw20\brdrcf2 \clpadt20 \clpadl100 \clpadb20 \clpadr100 \gaph\cellx2880
\clvertalt \clshdrawnil \clbrdrt\brdrs\brdrw20\brdrcf2 \clbrdrl\brdrs\brdrw20\brdrcf2 \clbrdrb\brdrs\brdrw20\brdrcf2 \clbrdrr\brdrs\brdrw20\brdrcf2 \clpadt20 \clpadl100 \clpadb20 \clpadr100 \gaph\cellx5760
\clvertalt \clshdrawnil \clbrdrt\brdrs\brdrw20\brdrcf2 \clbrdrl\brdrs\brdrw20\brdrcf2 \clbrdrb\brdrs\brdrw20\brdrcf2 \clbrdrr\brdrs\brdrw20\brdrcf2 \clpadt20 \clpadl100 \clpadb20 \clpadr100 \gaph\cellx8640
\pard\intbl\itap1\pardeftab560\slleading20\partightenfactor0

\f0\b\fs26 \cf0 2
\f3\b0\fs24 \cell 
\pard\intbl\itap1\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Defect
\f3\fs24 \cell 
\pard\intbl\itap1\pardeftab560\slleading20\partightenfactor0

\f0\b\fs26 \cf0 Dark Navy
\f1\b0  (Anomaly Feature)\cell \lastrow\row
\pard\pardeftab560\slleading20\pardirnatural\partightenfactor0
\cf0 \
}