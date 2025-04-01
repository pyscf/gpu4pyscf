/*
 * Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#define NGAUSS  (128)

 __device__
 static double r128[] = {
    2.0974980641241814e-09,6.706309141346622e-08,5.085417732475506e-07,2.1387593238664238e-06,6.510414150562305e-06,1.6149789330621722e-05,3.477832959797311e-05,6.752008446297975e-05,0.00012109262033321855,0.00020397925093973424,0.00032658072419877726,0.0005013448306671053,0.0007428727579217309,0.0010680013989176063,0.0014958612217488287,0.002047909713382867,0.002747940810594929,0.003622071117120762,0.00469870406710482,0.006008473522279845,0.007584168576217687,0.009460641577370588,0.011674701569110102,0.014264995477186626,0.017271879452467997,0.02073728280084053,0.02470456690574374,0.02921838147641287,0.03432452034205358,0.04006977886525154,0.046501814873751934,0.05366901481541042,0.06162036663348358,0.07040534064514625,0.08007377949115657,0.09067579801435666,0.10226169372367844,0.11488186831231562,0.1285867605265284,0.14342679052728535,0.15945231575189334,0.17671359816755083,0.19526078271344027,0.21514388665208117,0.23641279949339633,0.25911729311512466,0.2833070416795892,0.309031650937876,0.33634069651677867,0.3652837707998631,0.3959105380403516,0.42827079737883533,0.46241455348193694,0.49839209456791367,0.5362540776409215,0.5760516208165536,0.6178364026868168,0.6616607687425191,0.7075778449450105,0.7556416586173124,0.8059072669071227,0.8584308931613454,0.913270071644287,0.9704838011301781,1.0301327080062732,1.0922792196365685,1.156987748859655,1.224324890629046,1.2943596319525263,1.3671635764510053,1.44281118503981,1.521380034439527,1.6029510954532769,1.6876090332070943,1.7754425318452318,1.866544646508953,1.961013185813167,2.0589511284788973,2.160467078291771,2.265675762149974,2.3746985766549473,2.48766418950282,2.6047092028767747,2.725978887147429,2.85162799449364,2.9818216636007233,3.11673642842822,3.2565613462276923,3.4014992626117246,3.5517682346264867,3.707603136586523,3.8692574780488527,4.037005468934676,4.211144373707828,4.391997205021157,4.579915817775563,4.775284477664755,4.9785239947435755,5.190096533344465,5.410511236095016,5.64033083364239,5.880179455400774,6.130751913555642,6.392824807351983,6.667269893935096,6.9550703050790785,7.257340369576619,7.575350048736515,7.9105553369049835,8.264636464955368,8.639546441273588,9.037573480435166,9.461422378991507,9.914322187927917,10.400171088628467,10.923735051317658,11.490926170784794,12.109202398673274,12.788158334254895,13.540428351871086,14.383123898536038,15.340235594798905,16.446899576771763,17.757588876204775,19.363561867479117,21.435897004921358,24.358672177861372,29.35744981829575
 };

 __device__
static double w128[] = {1.0486305131668069e-08,1.675819697896843e-07,8.467079823364532e-07,2.6686184085191868e-06,6.492039268777722e-06,1.340348876948803e-05,2.4704413010691534e-05,4.1895816165119285e-05,6.666060941573645e-05,0.0001008439468332738,0.00014643185697929895,0.00020552851447737426,0.00028033252439314636,0.00037311261254195947,0.00048618312567858695,0.0006218797462083349,0.0007825358163391982,0.0009704596466912716,0.001187913154972045,0.0014370921425114894,0.0017201084716759567,0.0020389743571549565,0.0023955889307140147,0.0027917271841643035,0.0032290313409082732,0.0037090046542392597,0.004233007582150805,0.004802256245027731,0.005417823035219303,0.006080639216785492,0.006791499329991374,0.007551067198433777,0.008359883326785708,0.009218373473582337,0.010126858185621515,0.011085563087662984,0.012094629732363681,0.013154126829935883,0.014264061694025248,0.015424391758988555,0.01663503604338667,0.017895886454461413,0.019206818848100164,0.020567703777870573,0.021978416884791168,0.023438848896348237,0.02494891521871632,0.026508565120105788,0.02811779051563093,0.029776634375101228,0.03148519878477058,0.03324365270244722,0.0350522394526129,0.03691128401448268,0.03882120016142324,0.04078249751501977,0.04279578858151562,0.04486179584252773,0.04698135897604822,0.04915544228795644,0.05138514243877189,0.05367169655536073,0.05601649082295775,0.058421069659380494,0.060887145580907104,0.06341660987819053,0.06601154423104884,0.06867423340328631,0.07140717917319175,0.07421311567240672,0.0770950263258954,0.08005616260930208,0.08310006486766405,0.08623058547199673,0.08945191462856047,0.09276860920072326,0.09618562495653638,0.09970835271800149,0.10334265896244504,0.10709493151475627,0.11097213107437287,0.1149818494463566,0.11913237549609032,0.12343277002751835,0.1278929510022999,0.13252379078033674,0.13733722738172896,0.1423463921600684,0.14756575675465905,0.15301130277724145,0.1587007184161284,0.16465362704498734,0.17089185405394694,0.17743973954190545,0.184324506306365,0.19157669485477874,0.19923068009255288,0.20732528812494053,0.2159045365268794,0.22501852787750137,0.23472453486962172,0.24508832665599795,0.25618580137490254,0.2681050105794978,0.2809486898705686,0.2948374497888911,0.30991383704420494,0.3263475561875548,0.34434225787649264,0.36414447089376234,0.38605551159576557,0.4104475967860226,0.43778599900665927,0.46866006403803295,0.5038275217048119,0.5442792479177258,0.5913364078334866,0.6468005862919275,0.7131939966238761,0.7941598287707732,0.8951628586180937,1.024790579743745,1.1973554670550892,1.4386199570054774,1.8001088982463043,2.4020469038920345,3.6051077606163586,7.212660353196581
};


