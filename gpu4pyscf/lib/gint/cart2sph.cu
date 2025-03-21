/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
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

#include <stdio.h>

template<int L>
__device__
static void cart2sph(const double *gcart, double *gsph, const int cart_stride)
{
    if constexpr (L == 0) {
        gsph[0] = gcart[              0];
        // gsph[0] = 0.282094791773878143 * gcart[              0];
    } else if constexpr (L == 1) {
        gsph[0] = gcart[              0];
        gsph[1] = gcart[    cart_stride];
        gsph[2] = gcart[2 * cart_stride];
        // gsph[0] = 0.488602511902919921 * gcart[              0];
        // gsph[1] = 0.488602511902919921 * gcart[    cart_stride];
        // gsph[2] = 0.488602511902919921 * gcart[2 * cart_stride];
    } else if constexpr (L == 2) {
        gsph[0] = 1.092548430592079070 * gcart[    cart_stride];
        gsph[1] = 1.092548430592079070 * gcart[4 * cart_stride];
        gsph[2] = 0.630783130505040012 * gcart[5 * cart_stride] - 0.315391565252520002 * (gcart[              0] + gcart[3 * cart_stride]);
        gsph[3] = 1.092548430592079070 * gcart[2 * cart_stride];
        gsph[4] = 0.546274215296039535 * (gcart[              0] - gcart[3 * cart_stride]);
    } else if constexpr (L == 3) {
        gsph[0] = 0.5900435899266435 * (3 * gcart[    cart_stride] - gcart[6 * cart_stride]);
        gsph[1] = 2.8906114426405543 * gcart[4 * cart_stride];
        gsph[2] = 0.4570457994644657 * (4 * gcart[8 * cart_stride] - (gcart[    cart_stride] + gcart[6 * cart_stride]));
        gsph[3] = 1.1195289977703462 * ( - (gcart[2 * cart_stride] + gcart[7 * cart_stride]) + (2.0/3.0) * gcart[9 * cart_stride] );
        gsph[4] = 0.4570457994644657 * (4 * gcart[5 * cart_stride] - (gcart[              0] + gcart[3 * cart_stride]));
        gsph[5] = 1.4453057213202771 * (gcart[2 * cart_stride] - gcart[7 * cart_stride]);
        gsph[6] = 0.5900435899266435 * (gcart[              0] - 3 * gcart[3 * cart_stride]);
    } else if constexpr (L == 4) {
        gsph[0] = 2.5033429417967046 * (gcart[    cart_stride] - gcart[6 * cart_stride]);
        gsph[1] = 1.7701307697799304 * (3 * gcart[4 * cart_stride] - gcart[11 * cart_stride]);
        gsph[2] = 0.94617469575756 * (6 * gcart[8 * cart_stride] - gcart[    cart_stride] - gcart[6 * cart_stride]);
        gsph[3] = -2.0071396306718676 * (gcart[4 * cart_stride] + gcart[11 * cart_stride])
                + 2.676186174229157 * gcart[13 * cart_stride];
        gsph[4] = 0.31735664074561293 * (gcart[              0] + 2 * gcart[3 * cart_stride] + gcart[10 * cart_stride])
                - 2.5388531259649034 * (gcart[5 * cart_stride] + gcart[12 * cart_stride])
                + 0.8462843753216345 * gcart[14 * cart_stride];
        gsph[5] = -2.0071396306718676 * (gcart[2 * cart_stride] + gcart[7 * cart_stride])
                + 2.676186174229157 * gcart[9 * cart_stride];
        gsph[6] = 0.47308734787878 * ( - gcart[              0] + gcart[10 * cart_stride] + 6 * gcart[5 * cart_stride] - 6 * gcart[12 * cart_stride] );
        gsph[7] = 1.7701307697799304 * (gcart[2 * cart_stride] - 3 * gcart[7 * cart_stride]);
        gsph[8] = 0.6258357354491761 * (gcart[              0] - 6 * gcart[3 * cart_stride] + gcart[10 * cart_stride]);
    } else if constexpr (L == 5) {
        gsph[0] = 3.2819102842008507*gcart[    cart_stride] + -6.563820568401701*gcart[6 * cart_stride] + 0.6563820568401701*gcart[15 * cart_stride];
        gsph[1] = 8.302649259524165*gcart[4 * cart_stride] + -8.302649259524165*gcart[11 * cart_stride];
        gsph[2] = -1.467714898305751*gcart[    cart_stride] + -0.9784765988705008*gcart[6 * cart_stride] + 11.741719186446009*gcart[8 * cart_stride] + 0.4892382994352504*gcart[15 * cart_stride] + -3.913906395482003*gcart[17 * cart_stride];
        gsph[3] = -4.793536784973324*gcart[4 * cart_stride] + -4.793536784973324*gcart[11 * cart_stride] + 9.587073569946648*gcart[13 * cart_stride];
        gsph[4] = 0.45294665119569694*gcart[    cart_stride] + 0.9058933023913939*gcart[6 * cart_stride] + -5.435359814348363*gcart[8 * cart_stride] + 0.45294665119569694*gcart[15 * cart_stride] + -5.435359814348363*gcart[17 * cart_stride] + 3.6235732095655755*gcart[19 * cart_stride];
        gsph[5] = 1.754254836801354*gcart[2 * cart_stride] + 3.508509673602708*gcart[7 * cart_stride] + -4.678012898136944*gcart[9 * cart_stride] + 1.754254836801354*gcart[16 * cart_stride] + -4.678012898136944*gcart[18 * cart_stride] + 0.9356025796273888*gcart[20 * cart_stride];
        gsph[6] = 0.45294665119569694*gcart[              0] + 0.9058933023913939*gcart[3 * cart_stride] + -5.435359814348363*gcart[5 * cart_stride] + 0.45294665119569694*gcart[10 * cart_stride] + -5.435359814348363*gcart[12 * cart_stride] + 3.6235732095655755*gcart[14 * cart_stride];
        gsph[7] = -2.396768392486662*gcart[2 * cart_stride] + 4.793536784973324*gcart[9 * cart_stride] + 2.396768392486662*gcart[16 * cart_stride] + -4.793536784973324*gcart[18 * cart_stride];
        gsph[8] = -0.4892382994352504*gcart[              0] + 0.9784765988705008*gcart[3 * cart_stride] + 3.913906395482003*gcart[5 * cart_stride] + 1.467714898305751*gcart[10 * cart_stride] + -11.741719186446009*gcart[12 * cart_stride];
        gsph[9] = 2.075662314881041*gcart[2 * cart_stride] + -12.453973889286248*gcart[7 * cart_stride] + 2.075662314881041*gcart[16 * cart_stride];
        gsph[10] = 0.6563820568401701*gcart[              0] + -6.563820568401701*gcart[3 * cart_stride] + 3.2819102842008507*gcart[10 * cart_stride];
    } else if constexpr (L == 6) {
        gsph[0] = 4.099104631151486*gcart[    cart_stride] + -13.663682103838289*gcart[6 * cart_stride] + 4.099104631151486*gcart[15 * cart_stride];
        gsph[1] = 11.833095811158763*gcart[4 * cart_stride] + -23.666191622317527*gcart[11 * cart_stride] + 2.3666191622317525*gcart[22 * cart_stride];
        gsph[2] = -2.0182596029148963*gcart[    cart_stride] + 20.182596029148968*gcart[8 * cart_stride] + 2.0182596029148963*gcart[15 * cart_stride] + -20.182596029148968*gcart[17 * cart_stride];
        gsph[3] = -8.29084733563431*gcart[4 * cart_stride] + -5.527231557089541*gcart[11 * cart_stride] + 22.108926228358165*gcart[13 * cart_stride] + 2.7636157785447706*gcart[22 * cart_stride] + -7.369642076119389*gcart[24 * cart_stride];
        gsph[4] = 0.9212052595149236*gcart[    cart_stride] + 1.8424105190298472*gcart[6 * cart_stride] + -14.739284152238778*gcart[8 * cart_stride] + 0.9212052595149236*gcart[15 * cart_stride] + -14.739284152238778*gcart[17 * cart_stride] + 14.739284152238778*gcart[19 * cart_stride];
        gsph[5] = 2.913106812593657*gcart[4 * cart_stride] + 5.826213625187314*gcart[11 * cart_stride] + -11.652427250374627*gcart[13 * cart_stride] + 2.913106812593657*gcart[22 * cart_stride] + -11.652427250374627*gcart[24 * cart_stride] + 4.6609709001498505*gcart[26 * cart_stride];
        gsph[6] = -0.3178460113381421*gcart[              0] + -0.9535380340144264*gcart[3 * cart_stride] + 5.721228204086558*gcart[5 * cart_stride] + -0.9535380340144264*gcart[10 * cart_stride] + 11.442456408173117*gcart[12 * cart_stride] + -7.628304272115411*gcart[14 * cart_stride] + -0.3178460113381421*gcart[21 * cart_stride] + 5.721228204086558*gcart[23 * cart_stride] + -7.628304272115411*gcart[25 * cart_stride] + 1.0171072362820548*gcart[27 * cart_stride];
        gsph[7] = 2.913106812593657*gcart[2 * cart_stride] + 5.826213625187314*gcart[7 * cart_stride] + -11.652427250374627*gcart[9 * cart_stride] + 2.913106812593657*gcart[16 * cart_stride] + -11.652427250374627*gcart[18 * cart_stride] + 4.6609709001498505*gcart[20 * cart_stride];
        gsph[8] = 0.4606026297574618*gcart[              0] + 0.4606026297574618*gcart[3 * cart_stride] + -7.369642076119389*gcart[5 * cart_stride] + -0.4606026297574618*gcart[10 * cart_stride] + 7.369642076119389*gcart[14 * cart_stride] + -0.4606026297574618*gcart[21 * cart_stride] + 7.369642076119389*gcart[23 * cart_stride] + -7.369642076119389*gcart[25 * cart_stride];
        gsph[9] = -2.7636157785447706*gcart[2 * cart_stride] + 5.527231557089541*gcart[7 * cart_stride] + 7.369642076119389*gcart[9 * cart_stride] + 8.29084733563431*gcart[16 * cart_stride] + -22.108926228358165*gcart[18 * cart_stride];
        gsph[10] = -0.5045649007287241*gcart[              0] + 2.52282450364362*gcart[3 * cart_stride] + 5.045649007287242*gcart[5 * cart_stride] + 2.52282450364362*gcart[10 * cart_stride] + -30.273894043723452*gcart[12 * cart_stride] + -0.5045649007287241*gcart[21 * cart_stride] + 5.045649007287242*gcart[23 * cart_stride];
        gsph[11] = 2.3666191622317525*gcart[2 * cart_stride] + -23.666191622317527*gcart[7 * cart_stride] + 11.833095811158763*gcart[16 * cart_stride];
        gsph[12] = 0.6831841051919144*gcart[              0] + -10.247761577878716*gcart[3 * cart_stride] + 10.247761577878716*gcart[10 * cart_stride] + -0.6831841051919144*gcart[21 * cart_stride];
    } else if constexpr (L == 7) {
        gsph[0] = 4.950139127672174*gcart[    cart_stride] + -24.75069563836087*gcart[6 * cart_stride] + 14.850417383016522*gcart[15 * cart_stride] + -0.7071627325245963*gcart[28 * cart_stride];
        gsph[1] = 15.8757639708114*gcart[4 * cart_stride] + -52.919213236038004*gcart[11 * cart_stride] + 15.8757639708114*gcart[22 * cart_stride];
        gsph[2] = -2.594577893601302*gcart[    cart_stride] + 2.594577893601302*gcart[6 * cart_stride] + 31.134934723215622*gcart[8 * cart_stride] + 4.670240208482344*gcart[15 * cart_stride] + -62.269869446431244*gcart[17 * cart_stride] + -0.5189155787202604*gcart[28 * cart_stride] + 6.226986944643125*gcart[30 * cart_stride];
        gsph[3] = -12.45397388928625*gcart[4 * cart_stride] + 41.51324629762083*gcart[13 * cart_stride] + 12.45397388928625*gcart[22 * cart_stride] + -41.51324629762083*gcart[24 * cart_stride];
        gsph[4] = 1.4081304047606462*gcart[    cart_stride] + 2.3468840079344107*gcart[6 * cart_stride] + -28.162608095212924*gcart[8 * cart_stride] + 0.4693768015868821*gcart[15 * cart_stride] + -18.77507206347528*gcart[17 * cart_stride] + 37.55014412695057*gcart[19 * cart_stride] + -0.4693768015868821*gcart[28 * cart_stride] + 9.38753603173764*gcart[30 * cart_stride] + -12.516714708983523*gcart[32 * cart_stride];
        gsph[5] = 6.637990386674741*gcart[4 * cart_stride] + 13.275980773349483*gcart[11 * cart_stride] + -35.402615395598616*gcart[13 * cart_stride] + 6.637990386674741*gcart[22 * cart_stride] + -35.402615395598616*gcart[24 * cart_stride] + 21.241569237359172*gcart[26 * cart_stride];
        gsph[6] = -0.4516580379125866*gcart[    cart_stride] + -1.35497411373776*gcart[6 * cart_stride] + 10.839792909902078*gcart[8 * cart_stride] + -1.35497411373776*gcart[15 * cart_stride] + 21.679585819804156*gcart[17 * cart_stride] + -21.679585819804156*gcart[19 * cart_stride] + -0.4516580379125866*gcart[28 * cart_stride] + 10.839792909902078*gcart[30 * cart_stride] + -21.679585819804156*gcart[32 * cart_stride] + 5.781222885281109*gcart[34 * cart_stride];
        gsph[7] = -2.389949691920173*gcart[2 * cart_stride] + -7.169849075760519*gcart[7 * cart_stride] + 14.339698151521036*gcart[9 * cart_stride] + -7.169849075760519*gcart[16 * cart_stride] + 28.679396303042072*gcart[18 * cart_stride] + -11.47175852121683*gcart[20 * cart_stride] + -2.389949691920173*gcart[29 * cart_stride] + 14.339698151521036*gcart[31 * cart_stride] + -11.47175852121683*gcart[33 * cart_stride] + 1.092548430592079*gcart[35 * cart_stride];
        gsph[8] = -0.4516580379125866*gcart[              0] + -1.35497411373776*gcart[3 * cart_stride] + 10.839792909902078*gcart[5 * cart_stride] + -1.35497411373776*gcart[10 * cart_stride] + 21.679585819804156*gcart[12 * cart_stride] + -21.679585819804156*gcart[14 * cart_stride] + -0.4516580379125866*gcart[21 * cart_stride] + 10.839792909902078*gcart[23 * cart_stride] + -21.679585819804156*gcart[25 * cart_stride] + 5.781222885281109*gcart[27 * cart_stride];
        gsph[9] = 3.3189951933373707*gcart[2 * cart_stride] + 3.3189951933373707*gcart[7 * cart_stride] + -17.701307697799308*gcart[9 * cart_stride] + -3.3189951933373707*gcart[16 * cart_stride] + 10.620784618679586*gcart[20 * cart_stride] + -3.3189951933373707*gcart[29 * cart_stride] + 17.701307697799308*gcart[31 * cart_stride] + -10.620784618679586*gcart[33 * cart_stride];
        gsph[10] = 0.4693768015868821*gcart[              0] + -0.4693768015868821*gcart[3 * cart_stride] + -9.38753603173764*gcart[5 * cart_stride] + -2.3468840079344107*gcart[10 * cart_stride] + 18.77507206347528*gcart[12 * cart_stride] + 12.516714708983523*gcart[14 * cart_stride] + -1.4081304047606462*gcart[21 * cart_stride] + 28.162608095212924*gcart[23 * cart_stride] + -37.55014412695057*gcart[25 * cart_stride];
        gsph[11] = -3.1134934723215624*gcart[2 * cart_stride] + 15.567467361607811*gcart[7 * cart_stride] + 10.378311574405208*gcart[9 * cart_stride] + 15.567467361607811*gcart[16 * cart_stride] + -62.269869446431244*gcart[18 * cart_stride] + -3.1134934723215624*gcart[29 * cart_stride] + 10.378311574405208*gcart[31 * cart_stride];
        gsph[12] = -0.5189155787202604*gcart[              0] + 4.670240208482344*gcart[3 * cart_stride] + 6.226986944643125*gcart[5 * cart_stride] + 2.594577893601302*gcart[10 * cart_stride] + -62.269869446431244*gcart[12 * cart_stride] + -2.594577893601302*gcart[21 * cart_stride] + 31.134934723215622*gcart[23 * cart_stride];
        gsph[13] = 2.6459606618019*gcart[2 * cart_stride] + -39.6894099270285*gcart[7 * cart_stride] + 39.6894099270285*gcart[16 * cart_stride] + -2.6459606618019*gcart[29 * cart_stride];
        gsph[14] = 0.7071627325245963*gcart[              0] + -14.850417383016522*gcart[3 * cart_stride] + 24.75069563836087*gcart[10 * cart_stride] + -4.950139127672174*gcart[21 * cart_stride];
    } else if constexpr (L == 8) {
        gsph[0] = 5.83141328139864*gcart[    cart_stride] + -40.81989296979048*gcart[6 * cart_stride] + 40.81989296979048*gcart[15 * cart_stride] + -5.83141328139864*gcart[28 * cart_stride];
        gsph[1] = 20.40994648489524*gcart[4 * cart_stride] + -102.0497324244762*gcart[11 * cart_stride] + 61.22983945468572*gcart[22 * cart_stride] + -2.91570664069932*gcart[37 * cart_stride];
        gsph[2] = -3.193996596357255*gcart[    cart_stride] + 7.452658724833595*gcart[6 * cart_stride] + 44.71595234900157*gcart[8 * cart_stride] + 7.452658724833595*gcart[15 * cart_stride] + -149.0531744966719*gcart[17 * cart_stride] + -3.193996596357255*gcart[28 * cart_stride] + 44.71595234900157*gcart[30 * cart_stride];
        gsph[3] = -17.24955311049054*gcart[4 * cart_stride] + 17.24955311049054*gcart[11 * cart_stride] + 68.99821244196217*gcart[13 * cart_stride] + 31.04919559888297*gcart[22 * cart_stride] + -137.9964248839243*gcart[24 * cart_stride] + -3.449910622098108*gcart[37 * cart_stride] + 13.79964248839243*gcart[39 * cart_stride];
        gsph[4] = 1.913666099037323*gcart[    cart_stride] + 1.913666099037323*gcart[6 * cart_stride] + -45.92798637689575*gcart[8 * cart_stride] + -1.913666099037323*gcart[15 * cart_stride] + 76.54664396149292*gcart[19 * cart_stride] + -1.913666099037323*gcart[28 * cart_stride] + 45.92798637689575*gcart[30 * cart_stride] + -76.54664396149292*gcart[32 * cart_stride];
        gsph[5] = 11.1173953976599*gcart[4 * cart_stride] + 18.52899232943316*gcart[11 * cart_stride] + -74.11596931773265*gcart[13 * cart_stride] + 3.705798465886632*gcart[22 * cart_stride] + -49.41064621182176*gcart[24 * cart_stride] + 59.29277545418611*gcart[26 * cart_stride] + -3.705798465886632*gcart[37 * cart_stride] + 24.70532310591088*gcart[39 * cart_stride] + -19.7642584847287*gcart[41 * cart_stride];
        gsph[6] = -0.912304516869819*gcart[    cart_stride] + -2.736913550609457*gcart[6 * cart_stride] + 27.36913550609457*gcart[8 * cart_stride] + -2.736913550609457*gcart[15 * cart_stride] + 54.73827101218914*gcart[17 * cart_stride] + -72.98436134958553*gcart[19 * cart_stride] + -0.912304516869819*gcart[28 * cart_stride] + 27.36913550609457*gcart[30 * cart_stride] + -72.98436134958553*gcart[32 * cart_stride] + 29.19374453983421*gcart[34 * cart_stride];
        gsph[7] = -3.8164436064573*gcart[4 * cart_stride] + -11.4493308193719*gcart[11 * cart_stride] + 30.5315488516584*gcart[13 * cart_stride] + -11.4493308193719*gcart[22 * cart_stride] + 61.06309770331679*gcart[24 * cart_stride] + -36.63785862199007*gcart[26 * cart_stride] + -3.8164436064573*gcart[37 * cart_stride] + 30.5315488516584*gcart[39 * cart_stride] + -36.63785862199007*gcart[41 * cart_stride] + 6.978639737521918*gcart[43 * cart_stride];
        gsph[8] = 0.3180369672047749*gcart[              0] + 1.272147868819099*gcart[3 * cart_stride] + -10.1771829505528*gcart[5 * cart_stride] + 1.908221803228649*gcart[10 * cart_stride] + -30.53154885165839*gcart[12 * cart_stride] + 30.53154885165839*gcart[14 * cart_stride] + 1.272147868819099*gcart[21 * cart_stride] + -30.53154885165839*gcart[23 * cart_stride] + 61.06309770331677*gcart[25 * cart_stride] + -16.28349272088447*gcart[27 * cart_stride] + 0.3180369672047749*gcart[36 * cart_stride] + -10.1771829505528*gcart[38 * cart_stride] + 30.53154885165839*gcart[40 * cart_stride] + -16.28349272088447*gcart[42 * cart_stride] + 1.16310662292032*gcart[44 * cart_stride];
        gsph[9] = -3.8164436064573*gcart[2 * cart_stride] + -11.4493308193719*gcart[7 * cart_stride] + 30.5315488516584*gcart[9 * cart_stride] + -11.4493308193719*gcart[16 * cart_stride] + 61.06309770331679*gcart[18 * cart_stride] + -36.63785862199007*gcart[20 * cart_stride] + -3.8164436064573*gcart[29 * cart_stride] + 30.5315488516584*gcart[31 * cart_stride] + -36.63785862199007*gcart[33 * cart_stride] + 6.978639737521918*gcart[35 * cart_stride];
        gsph[10] = -0.4561522584349095*gcart[              0] + -0.912304516869819*gcart[3 * cart_stride] + 13.68456775304729*gcart[5 * cart_stride] + 13.68456775304729*gcart[12 * cart_stride] + -36.49218067479276*gcart[14 * cart_stride] + 0.912304516869819*gcart[21 * cart_stride] + -13.68456775304729*gcart[23 * cart_stride] + 14.5968722699171*gcart[27 * cart_stride] + 0.4561522584349095*gcart[36 * cart_stride] + -13.68456775304729*gcart[38 * cart_stride] + 36.49218067479276*gcart[40 * cart_stride] + -14.5968722699171*gcart[42 * cart_stride];
        gsph[11] = 3.705798465886632*gcart[2 * cart_stride] + -3.705798465886632*gcart[7 * cart_stride] + -24.70532310591088*gcart[9 * cart_stride] + -18.52899232943316*gcart[16 * cart_stride] + 49.41064621182176*gcart[18 * cart_stride] + 19.7642584847287*gcart[20 * cart_stride] + -11.1173953976599*gcart[29 * cart_stride] + 74.11596931773265*gcart[31 * cart_stride] + -59.29277545418611*gcart[33 * cart_stride];
        gsph[12] = 0.4784165247593308*gcart[              0] + -1.913666099037323*gcart[3 * cart_stride] + -11.48199659422394*gcart[5 * cart_stride] + -4.784165247593307*gcart[10 * cart_stride] + 57.40998297111968*gcart[12 * cart_stride] + 19.13666099037323*gcart[14 * cart_stride] + -1.913666099037323*gcart[21 * cart_stride] + 57.40998297111968*gcart[23 * cart_stride] + -114.8199659422394*gcart[25 * cart_stride] + 0.4784165247593308*gcart[36 * cart_stride] + -11.48199659422394*gcart[38 * cart_stride] + 19.13666099037323*gcart[40 * cart_stride];
        gsph[13] = -3.449910622098108*gcart[2 * cart_stride] + 31.04919559888297*gcart[7 * cart_stride] + 13.79964248839243*gcart[9 * cart_stride] + 17.24955311049054*gcart[16 * cart_stride] + -137.9964248839243*gcart[18 * cart_stride] + -17.24955311049054*gcart[29 * cart_stride] + 68.99821244196217*gcart[31 * cart_stride];
        gsph[14] = -0.5323327660595425*gcart[              0] + 7.452658724833595*gcart[3 * cart_stride] + 7.452658724833595*gcart[5 * cart_stride] + -111.7898808725039*gcart[12 * cart_stride] + -7.452658724833595*gcart[21 * cart_stride] + 111.7898808725039*gcart[23 * cart_stride] + 0.5323327660595425*gcart[36 * cart_stride] + -7.452658724833595*gcart[38 * cart_stride];
        gsph[15] = 2.91570664069932*gcart[2 * cart_stride] + -61.22983945468572*gcart[7 * cart_stride] + 102.0497324244762*gcart[16 * cart_stride] + -20.40994648489524*gcart[29 * cart_stride];
        gsph[16] = 0.72892666017483*gcart[              0] + -20.40994648489524*gcart[3 * cart_stride] + 51.0248662122381*gcart[10 * cart_stride] + -20.40994648489524*gcart[21 * cart_stride] + 0.72892666017483*gcart[36 * cart_stride];
    } else if constexpr (L == 9) {
        gsph[0] = 6.740108566678694*gcart[    cart_stride] + -62.9076799556678*gcart[6 * cart_stride] + 94.36151993350171*gcart[15 * cart_stride] + -26.96043426671477*gcart[28 * cart_stride] + 0.7489009518531882*gcart[45 * cart_stride];
        gsph[1] = 25.41854119163758*gcart[4 * cart_stride] + -177.9297883414631*gcart[11 * cart_stride] + 177.9297883414631*gcart[22 * cart_stride] + -25.41854119163758*gcart[37 * cart_stride];
        gsph[2] = -3.814338369408373*gcart[    cart_stride] + 15.25735347763349*gcart[6 * cart_stride] + 61.02941391053396*gcart[8 * cart_stride] + 7.628676738816745*gcart[15 * cart_stride] + -305.1470695526698*gcart[17 * cart_stride] + -10.89810962688107*gcart[28 * cart_stride] + 183.0882417316019*gcart[30 * cart_stride] + 0.5449054813440533*gcart[45 * cart_stride] + -8.718487701504852*gcart[47 * cart_stride];
        gsph[3] = -22.65129549625621*gcart[4 * cart_stride] + 52.85302282459782*gcart[11 * cart_stride] + 105.7060456491956*gcart[13 * cart_stride] + 52.85302282459782*gcart[22 * cart_stride] + -352.3534854973187*gcart[24 * cart_stride] + -22.65129549625621*gcart[37 * cart_stride] + 105.7060456491956*gcart[39 * cart_stride];
        gsph[4] = 2.436891395195093*gcart[    cart_stride] + -68.23295906546261*gcart[8 * cart_stride] + -6.82329590654626*gcart[15 * cart_stride] + 68.23295906546261*gcart[17 * cart_stride] + 136.4659181309252*gcart[19 * cart_stride] + -3.899026232312149*gcart[28 * cart_stride] + 122.8193263178327*gcart[30 * cart_stride] + -272.9318362618504*gcart[32 * cart_stride] + 0.4873782790390186*gcart[45 * cart_stride] + -13.64659181309252*gcart[47 * cart_stride] + 27.29318362618504*gcart[49 * cart_stride];
        gsph[5] = 16.31079695491669*gcart[4 * cart_stride] + 16.31079695491669*gcart[11 * cart_stride] + -130.4863756393335*gcart[13 * cart_stride] + -16.31079695491669*gcart[22 * cart_stride] + 130.4863756393335*gcart[26 * cart_stride] + -16.31079695491669*gcart[37 * cart_stride] + 130.4863756393335*gcart[39 * cart_stride] + -130.4863756393335*gcart[41 * cart_stride];
        gsph[6] = -1.385125560048583*gcart[    cart_stride] + -3.693668160129556*gcart[6 * cart_stride] + 49.864520161749*gcart[8 * cart_stride] + -2.770251120097167*gcart[15 * cart_stride] + 83.107533602915*gcart[17 * cart_stride] + -166.21506720583*gcart[19 * cart_stride] + 16.621506720583*gcart[30 * cart_stride] + -110.8100448038867*gcart[32 * cart_stride] + 88.64803584310934*gcart[34 * cart_stride] + 0.4617085200161945*gcart[45 * cart_stride] + -16.621506720583*gcart[47 * cart_stride] + 55.40502240194333*gcart[49 * cart_stride] + -29.54934528103645*gcart[51 * cart_stride];
        gsph[7] = -8.46325696792098*gcart[4 * cart_stride] + -25.38977090376294*gcart[11 * cart_stride] + 84.6325696792098*gcart[13 * cart_stride] + -25.38977090376294*gcart[22 * cart_stride] + 169.2651393584196*gcart[24 * cart_stride] + -135.4121114867357*gcart[26 * cart_stride] + -8.46325696792098*gcart[37 * cart_stride] + 84.6325696792098*gcart[39 * cart_stride] + -135.4121114867357*gcart[41 * cart_stride] + 38.68917471049591*gcart[43 * cart_stride];
        gsph[8] = 0.451093112065591*gcart[    cart_stride] + 1.804372448262364*gcart[6 * cart_stride] + -18.04372448262364*gcart[8 * cart_stride] + 2.706558672393546*gcart[15 * cart_stride] + -54.13117344787092*gcart[17 * cart_stride] + 72.17489793049457*gcart[19 * cart_stride] + 1.804372448262364*gcart[28 * cart_stride] + -54.13117344787092*gcart[30 * cart_stride] + 144.3497958609891*gcart[32 * cart_stride] + -57.73991834439565*gcart[34 * cart_stride] + 0.451093112065591*gcart[45 * cart_stride] + -18.04372448262364*gcart[47 * cart_stride] + 72.17489793049457*gcart[49 * cart_stride] + -57.73991834439565*gcart[51 * cart_stride] + 8.248559763485094*gcart[53 * cart_stride];
        gsph[9] = 3.026024588281776*gcart[2 * cart_stride] + 12.1040983531271*gcart[7 * cart_stride] + -32.27759560833895*gcart[9 * cart_stride] + 18.15614752969066*gcart[16 * cart_stride] + -96.83278682501685*gcart[18 * cart_stride] + 58.0996720950101*gcart[20 * cart_stride] + 12.1040983531271*gcart[29 * cart_stride] + -96.83278682501685*gcart[31 * cart_stride] + 116.1993441900202*gcart[33 * cart_stride] + -22.1332084171467*gcart[35 * cart_stride] + 3.026024588281776*gcart[46 * cart_stride] + -32.27759560833895*gcart[48 * cart_stride] + 58.0996720950101*gcart[50 * cart_stride] + -22.1332084171467*gcart[52 * cart_stride] + 1.229622689841484*gcart[54 * cart_stride];
        gsph[10] = 0.451093112065591*gcart[              0] + 1.804372448262364*gcart[3 * cart_stride] + -18.04372448262364*gcart[5 * cart_stride] + 2.706558672393546*gcart[10 * cart_stride] + -54.13117344787092*gcart[12 * cart_stride] + 72.17489793049457*gcart[14 * cart_stride] + 1.804372448262364*gcart[21 * cart_stride] + -54.13117344787092*gcart[23 * cart_stride] + 144.3497958609891*gcart[25 * cart_stride] + -57.73991834439565*gcart[27 * cart_stride] + 0.451093112065591*gcart[36 * cart_stride] + -18.04372448262364*gcart[38 * cart_stride] + 72.17489793049457*gcart[40 * cart_stride] + -57.73991834439565*gcart[42 * cart_stride] + 8.248559763485094*gcart[44 * cart_stride];
        gsph[11] = -4.23162848396049*gcart[2 * cart_stride] + -8.46325696792098*gcart[7 * cart_stride] + 42.3162848396049*gcart[9 * cart_stride] + 42.3162848396049*gcart[18 * cart_stride] + -67.70605574336784*gcart[20 * cart_stride] + 8.46325696792098*gcart[29 * cart_stride] + -42.3162848396049*gcart[31 * cart_stride] + 19.34458735524795*gcart[35 * cart_stride] + 4.23162848396049*gcart[46 * cart_stride] + -42.3162848396049*gcart[48 * cart_stride] + 67.70605574336784*gcart[50 * cart_stride] + -19.34458735524795*gcart[52 * cart_stride];
        gsph[12] = -0.4617085200161945*gcart[              0] + 16.621506720583*gcart[5 * cart_stride] + 2.770251120097167*gcart[10 * cart_stride] + -16.621506720583*gcart[12 * cart_stride] + -55.40502240194333*gcart[14 * cart_stride] + 3.693668160129556*gcart[21 * cart_stride] + -83.107533602915*gcart[23 * cart_stride] + 110.8100448038867*gcart[25 * cart_stride] + 29.54934528103645*gcart[27 * cart_stride] + 1.385125560048583*gcart[36 * cart_stride] + -49.864520161749*gcart[38 * cart_stride] + 166.21506720583*gcart[40 * cart_stride] + -88.64803584310934*gcart[42 * cart_stride];
        gsph[13] = 4.077699238729173*gcart[2 * cart_stride] + -16.31079695491669*gcart[7 * cart_stride] + -32.62159390983339*gcart[9 * cart_stride] + -40.77699238729173*gcart[16 * cart_stride] + 163.1079695491669*gcart[18 * cart_stride] + 32.62159390983339*gcart[20 * cart_stride] + -16.31079695491669*gcart[29 * cart_stride] + 163.1079695491669*gcart[31 * cart_stride] + -195.7295634590003*gcart[33 * cart_stride] + 4.077699238729173*gcart[46 * cart_stride] + -32.62159390983339*gcart[48 * cart_stride] + 32.62159390983339*gcart[50 * cart_stride];
        gsph[14] = 0.4873782790390186*gcart[              0] + -3.899026232312149*gcart[3 * cart_stride] + -13.64659181309252*gcart[5 * cart_stride] + -6.82329590654626*gcart[10 * cart_stride] + 122.8193263178327*gcart[12 * cart_stride] + 27.29318362618504*gcart[14 * cart_stride] + 68.23295906546261*gcart[23 * cart_stride] + -272.9318362618504*gcart[25 * cart_stride] + 2.436891395195093*gcart[36 * cart_stride] + -68.23295906546261*gcart[38 * cart_stride] + 136.4659181309252*gcart[40 * cart_stride];
        gsph[15] = -3.775215916042701*gcart[2 * cart_stride] + 52.85302282459782*gcart[7 * cart_stride] + 17.61767427486594*gcart[9 * cart_stride] + -264.2651141229891*gcart[18 * cart_stride] + -52.85302282459782*gcart[29 * cart_stride] + 264.2651141229891*gcart[31 * cart_stride] + 3.775215916042701*gcart[46 * cart_stride] + -17.61767427486594*gcart[48 * cart_stride];
        gsph[16] = -0.5449054813440533*gcart[              0] + 10.89810962688107*gcart[3 * cart_stride] + 8.718487701504852*gcart[5 * cart_stride] + -7.628676738816745*gcart[10 * cart_stride] + -183.0882417316019*gcart[12 * cart_stride] + -15.25735347763349*gcart[21 * cart_stride] + 305.1470695526698*gcart[23 * cart_stride] + 3.814338369408373*gcart[36 * cart_stride] + -61.02941391053396*gcart[38 * cart_stride];
        gsph[17] = 3.177317648954698*gcart[2 * cart_stride] + -88.96489417073154*gcart[7 * cart_stride] + 222.4122354268289*gcart[16 * cart_stride] + -88.96489417073154*gcart[29 * cart_stride] + 3.177317648954698*gcart[46 * cart_stride];
        gsph[18] = 0.7489009518531882*gcart[              0] + -26.96043426671477*gcart[3 * cart_stride] + 94.36151993350171*gcart[10 * cart_stride] + -62.9076799556678*gcart[21 * cart_stride] + 6.740108566678694*gcart[36 * cart_stride];
    } else if constexpr (L == 10) {
        gsph[0] = 7.673951182219901*gcart[    cart_stride] + -92.08741418663881*gcart[6 * cart_stride] + 193.3835697919415*gcart[15 * cart_stride] + -92.08741418663881*gcart[28 * cart_stride] + 7.673951182219901*gcart[45 * cart_stride];
        gsph[1] = 30.88705769902543*gcart[4 * cart_stride] + -288.2792051909041*gcart[11 * cart_stride] + 432.4188077863561*gcart[22 * cart_stride] + -123.5482307961017*gcart[37 * cart_stride] + 3.431895299891715*gcart[56 * cart_stride];
        gsph[2] = -4.453815461763347*gcart[    cart_stride] + 26.72289277058008*gcart[6 * cart_stride] + 80.16867831174027*gcart[8 * cart_stride] + -561.1807481821819*gcart[17 * cart_stride] + -26.72289277058008*gcart[28 * cart_stride] + 561.1807481821819*gcart[30 * cart_stride] + 4.453815461763347*gcart[45 * cart_stride] + -80.16867831174027*gcart[47 * cart_stride];
        gsph[3] = -28.63763513582592*gcart[4 * cart_stride] + 114.5505405433037*gcart[11 * cart_stride] + 152.7340540577382*gcart[13 * cart_stride] + 57.27527027165184*gcart[22 * cart_stride] + -763.6702702886912*gcart[24 * cart_stride] + -81.82181467378834*gcart[37 * cart_stride] + 458.2021621732147*gcart[39 * cart_stride] + 4.091090733689417*gcart[56 * cart_stride] + -21.81915057967689*gcart[58 * cart_stride];
        gsph[4] = 2.976705744527138*gcart[    cart_stride] + -3.968940992702851*gcart[6 * cart_stride] + -95.25458382486842*gcart[8 * cart_stride] + -13.89129347445998*gcart[15 * cart_stride] + 222.2606955913596*gcart[17 * cart_stride] + 222.2606955913597*gcart[19 * cart_stride] + -3.968940992702851*gcart[28 * cart_stride] + 222.2606955913596*gcart[30 * cart_stride] + -740.8689853045323*gcart[32 * cart_stride] + 2.976705744527138*gcart[45 * cart_stride] + -95.25458382486842*gcart[47 * cart_stride] + 222.2606955913597*gcart[49 * cart_stride];
        gsph[5] = 22.18705464592268*gcart[4 * cart_stride] + -207.0791766952783*gcart[13 * cart_stride] + -62.12375300858349*gcart[22 * cart_stride] + 207.0791766952783*gcart[24 * cart_stride] + 248.495012034334*gcart[26 * cart_stride] + -35.49928743347628*gcart[37 * cart_stride] + 372.742518051501*gcart[39 * cart_stride] + -496.990024068668*gcart[41 * cart_stride] + 4.437410929184535*gcart[56 * cart_stride] + -41.41583533905566*gcart[58 * cart_stride] + 49.6990024068668*gcart[60 * cart_stride];
        gsph[6] = -1.870976726712969*gcart[    cart_stride] + -3.741953453425937*gcart[6 * cart_stride] + 78.58102252194469*gcart[8 * cart_stride] + 78.58102252194469*gcart[17 * cart_stride] + -314.3240900877788*gcart[19 * cart_stride] + 3.741953453425937*gcart[28 * cart_stride] + -78.58102252194469*gcart[30 * cart_stride] + 209.5493933918525*gcart[34 * cart_stride] + 1.870976726712969*gcart[45 * cart_stride] + -78.58102252194469*gcart[47 * cart_stride] + 314.3240900877788*gcart[49 * cart_stride] + -209.5493933918525*gcart[51 * cart_stride];
        gsph[7] = -13.89129347445998*gcart[4 * cart_stride] + -37.04344926522661*gcart[11 * cart_stride] + 166.6955216935197*gcart[13 * cart_stride] + -27.78258694891996*gcart[22 * cart_stride] + 277.8258694891996*gcart[24 * cart_stride] + -333.3910433870395*gcart[26 * cart_stride] + 55.56517389783991*gcart[39 * cart_stride] + -222.2606955913596*gcart[41 * cart_stride] + 127.0061117664912*gcart[43 * cart_stride] + 4.630431158153326*gcart[56 * cart_stride] + -55.56517389783991*gcart[58 * cart_stride] + 111.1303477956798*gcart[60 * cart_stride] + -42.33537058883041*gcart[62 * cart_stride];
        gsph[8] = 0.9081022627604556*gcart[    cart_stride] + 3.632409051041822*gcart[6 * cart_stride] + -43.58890861250187*gcart[8 * cart_stride] + 5.448613576562733*gcart[15 * cart_stride] + -130.7667258375056*gcart[17 * cart_stride] + 217.9445430625093*gcart[19 * cart_stride] + 3.632409051041822*gcart[28 * cart_stride] + -130.7667258375056*gcart[30 * cart_stride] + 435.8890861250187*gcart[32 * cart_stride] + -232.4741792666766*gcart[34 * cart_stride] + 0.9081022627604556*gcart[45 * cart_stride] + -43.58890861250187*gcart[47 * cart_stride] + 217.9445430625093*gcart[49 * cart_stride] + -232.4741792666766*gcart[51 * cart_stride] + 49.815895557145*gcart[53 * cart_stride];
        gsph[9] = 4.718637772708116*gcart[4 * cart_stride] + 18.87455109083247*gcart[11 * cart_stride] + -62.91517030277488*gcart[13 * cart_stride] + 28.3118266362487*gcart[22 * cart_stride] + -188.7455109083247*gcart[24 * cart_stride] + 150.9964087266597*gcart[26 * cart_stride] + 18.87455109083247*gcart[37 * cart_stride] + -188.7455109083247*gcart[39 * cart_stride] + 301.9928174533194*gcart[41 * cart_stride] + -86.28366212951984*gcart[43 * cart_stride] + 4.718637772708116*gcart[56 * cart_stride] + -62.91517030277488*gcart[58 * cart_stride] + 150.9964087266597*gcart[60 * cart_stride] + -86.28366212951984*gcart[62 * cart_stride] + 9.587073569946648*gcart[64 * cart_stride];
        gsph[10] = -0.3181304937373671*gcart[              0] + -1.590652468686835*gcart[3 * cart_stride] + 15.90652468686835*gcart[5 * cart_stride] + -3.181304937373671*gcart[10 * cart_stride] + 63.62609874747341*gcart[12 * cart_stride] + -84.83479832996456*gcart[14 * cart_stride] + -3.181304937373671*gcart[21 * cart_stride] + 95.43914812121012*gcart[23 * cart_stride] + -254.5043949898937*gcart[25 * cart_stride] + 101.8017579959575*gcart[27 * cart_stride] + -1.590652468686835*gcart[36 * cart_stride] + 63.62609874747341*gcart[38 * cart_stride] + -254.5043949898937*gcart[40 * cart_stride] + 203.6035159919149*gcart[42 * cart_stride] + -29.08621657027356*gcart[44 * cart_stride] + -0.3181304937373671*gcart[55 * cart_stride] + 15.90652468686835*gcart[57 * cart_stride] + -84.83479832996456*gcart[59 * cart_stride] + 101.8017579959575*gcart[61 * cart_stride] + -29.08621657027356*gcart[63 * cart_stride] + 1.292720736456603*gcart[65 * cart_stride];
        gsph[11] = 4.718637772708116*gcart[2 * cart_stride] + 18.87455109083247*gcart[7 * cart_stride] + -62.91517030277488*gcart[9 * cart_stride] + 28.3118266362487*gcart[16 * cart_stride] + -188.7455109083247*gcart[18 * cart_stride] + 150.9964087266597*gcart[20 * cart_stride] + 18.87455109083247*gcart[29 * cart_stride] + -188.7455109083247*gcart[31 * cart_stride] + 301.9928174533194*gcart[33 * cart_stride] + -86.28366212951984*gcart[35 * cart_stride] + 4.718637772708116*gcart[46 * cart_stride] + -62.91517030277488*gcart[48 * cart_stride] + 150.9964087266597*gcart[50 * cart_stride] + -86.28366212951984*gcart[52 * cart_stride] + 9.587073569946648*gcart[54 * cart_stride];
        gsph[12] = 0.4540511313802278*gcart[              0] + 1.362153394140683*gcart[3 * cart_stride] + -21.79445430625093*gcart[5 * cart_stride] + 0.9081022627604556*gcart[10 * cart_stride] + -43.58890861250187*gcart[12 * cart_stride] + 108.9722715312547*gcart[14 * cart_stride] + -0.9081022627604556*gcart[21 * cart_stride] + 108.9722715312547*gcart[25 * cart_stride] + -116.2370896333383*gcart[27 * cart_stride] + -1.362153394140683*gcart[36 * cart_stride] + 43.58890861250187*gcart[38 * cart_stride] + -108.9722715312547*gcart[40 * cart_stride] + 24.9079477785725*gcart[44 * cart_stride] + -0.4540511313802278*gcart[55 * cart_stride] + 21.79445430625093*gcart[57 * cart_stride] + -108.9722715312547*gcart[59 * cart_stride] + 116.2370896333383*gcart[61 * cart_stride] + -24.9079477785725*gcart[63 * cart_stride];
        gsph[13] = -4.630431158153326*gcart[2 * cart_stride] + 55.56517389783991*gcart[9 * cart_stride] + 27.78258694891996*gcart[16 * cart_stride] + -55.56517389783991*gcart[18 * cart_stride] + -111.1303477956798*gcart[20 * cart_stride] + 37.04344926522661*gcart[29 * cart_stride] + -277.8258694891996*gcart[31 * cart_stride] + 222.2606955913596*gcart[33 * cart_stride] + 42.33537058883041*gcart[35 * cart_stride] + 13.89129347445998*gcart[46 * cart_stride] + -166.6955216935197*gcart[48 * cart_stride] + 333.3910433870395*gcart[50 * cart_stride] + -127.0061117664912*gcart[52 * cart_stride];
        gsph[14] = -0.4677441816782422*gcart[              0] + 1.403232545034726*gcart[3 * cart_stride] + 19.64525563048617*gcart[5 * cart_stride] + 6.548418543495391*gcart[10 * cart_stride] + -78.58102252194469*gcart[12 * cart_stride] + -78.58102252194469*gcart[14 * cart_stride] + 6.548418543495391*gcart[21 * cart_stride] + -196.4525563048617*gcart[23 * cart_stride] + 392.9051126097235*gcart[25 * cart_stride] + 52.38734834796313*gcart[27 * cart_stride] + 1.403232545034726*gcart[36 * cart_stride] + -78.58102252194469*gcart[38 * cart_stride] + 392.9051126097235*gcart[40 * cart_stride] + -314.3240900877788*gcart[42 * cart_stride] + -0.4677441816782422*gcart[55 * cart_stride] + 19.64525563048617*gcart[57 * cart_stride] + -78.58102252194469*gcart[59 * cart_stride] + 52.38734834796313*gcart[61 * cart_stride];
        gsph[15] = 4.437410929184535*gcart[2 * cart_stride] + -35.49928743347628*gcart[7 * cart_stride] + -41.41583533905566*gcart[9 * cart_stride] + -62.12375300858349*gcart[16 * cart_stride] + 372.742518051501*gcart[18 * cart_stride] + 49.6990024068668*gcart[20 * cart_stride] + 207.0791766952783*gcart[31 * cart_stride] + -496.990024068668*gcart[33 * cart_stride] + 22.18705464592268*gcart[46 * cart_stride] + -207.0791766952783*gcart[48 * cart_stride] + 248.495012034334*gcart[50 * cart_stride];
        gsph[16] = 0.4961176240878564*gcart[              0] + -6.449529113142133*gcart[3 * cart_stride] + -15.8757639708114*gcart[5 * cart_stride] + -6.945646737229989*gcart[10 * cart_stride] + 222.2606955913596*gcart[12 * cart_stride] + 37.04344926522661*gcart[14 * cart_stride] + 6.945646737229989*gcart[21 * cart_stride] + -555.6517389783992*gcart[25 * cart_stride] + 6.449529113142133*gcart[36 * cart_stride] + -222.2606955913596*gcart[38 * cart_stride] + 555.6517389783992*gcart[40 * cart_stride] + -0.4961176240878564*gcart[55 * cart_stride] + 15.8757639708114*gcart[57 * cart_stride] + -37.04344926522661*gcart[59 * cart_stride];
        gsph[17] = -4.091090733689417*gcart[2 * cart_stride] + 81.82181467378834*gcart[7 * cart_stride] + 21.81915057967689*gcart[9 * cart_stride] + -57.27527027165184*gcart[16 * cart_stride] + -458.2021621732147*gcart[18 * cart_stride] + -114.5505405433037*gcart[29 * cart_stride] + 763.6702702886912*gcart[31 * cart_stride] + 28.63763513582592*gcart[46 * cart_stride] + -152.7340540577382*gcart[48 * cart_stride];
        gsph[18] = -0.5567269327204184*gcart[              0] + 15.0316271834513*gcart[3 * cart_stride] + 10.02108478896753*gcart[5 * cart_stride] + -23.38253117425757*gcart[10 * cart_stride] + -280.590374091091*gcart[12 * cart_stride] + -23.38253117425757*gcart[21 * cart_stride] + 701.4759352277273*gcart[23 * cart_stride] + 15.0316271834513*gcart[36 * cart_stride] + -280.590374091091*gcart[38 * cart_stride] + -0.5567269327204184*gcart[55 * cart_stride] + 10.02108478896753*gcart[57 * cart_stride];
        gsph[19] = 3.431895299891715*gcart[2 * cart_stride] + -123.5482307961017*gcart[7 * cart_stride] + 432.4188077863561*gcart[16 * cart_stride] + -288.2792051909041*gcart[29 * cart_stride] + 30.88705769902543*gcart[46 * cart_stride];
        gsph[20] = 0.7673951182219901*gcart[              0] + -34.53278031998956*gcart[3 * cart_stride] + 161.1529748266179*gcart[10 * cart_stride] + -161.1529748266179*gcart[21 * cart_stride] + 34.53278031998956*gcart[36 * cart_stride] + -0.7673951182219901*gcart[55 * cart_stride];
    } else {
        gsph[0] = NAN;
    }
}

template<int L>
__device__
static void sph2cart(double *gcart, const double *gsph, const int cart_stride)
{
    if constexpr (L == 0) {
        gcart[              0] = gsph[0];
        // gcart[              0] = 0.282094791773878143 * gsph[0];
    } else if constexpr (L == 1) {
        gcart[              0] = gsph[0];
        gcart[    cart_stride] = gsph[1];
        gcart[2 * cart_stride] = gsph[2];
        // gcart[              0] = 0.488602511902919921 * gsph[0];
        // gcart[    cart_stride] = 0.488602511902919921 * gsph[1];
        // gcart[2 * cart_stride] = 0.488602511902919921 * gsph[2];
    } else if constexpr (L == 2) {
        gcart[              0] = -0.31539156525252*gsph[2] + 0.5462742152960396*gsph[4];
        gcart[    cart_stride] = 1.0925484305920792*gsph[0];
        gcart[2 * cart_stride] = 1.0925484305920792*gsph[3];
        gcart[3 * cart_stride] = -0.31539156525252*gsph[2] + -0.5462742152960396*gsph[4];
        gcart[4 * cart_stride] = 1.0925484305920792*gsph[1];
        gcart[5 * cart_stride] = 0.63078313050504*gsph[2];
    } else if constexpr (L == 3) {
        gcart[              0] = -0.4570457994644657*gsph[4] + 0.5900435899266435*gsph[6];
        gcart[    cart_stride] = 1.7701307697799304*gsph[0] + -0.4570457994644657*gsph[2];
        gcart[2 * cart_stride] = -1.1195289977703462*gsph[3] + 1.4453057213202771*gsph[5];
        gcart[3 * cart_stride] = -0.4570457994644657*gsph[4] + -1.7701307697799304*gsph[6];
        gcart[4 * cart_stride] = 2.8906114426405543*gsph[1];
        gcart[5 * cart_stride] = 1.8281831978578629*gsph[4];
        gcart[6 * cart_stride] = -0.5900435899266435*gsph[0] + -0.4570457994644657*gsph[2];
        gcart[7 * cart_stride] = -1.1195289977703462*gsph[3] + -1.4453057213202771*gsph[5];
        gcart[8 * cart_stride] = 1.8281831978578629*gsph[2];
        gcart[9 * cart_stride] = 0.7463526651802308*gsph[3];
    } else if constexpr (L == 4) {
        gcart[              0] = 0.31735664074561293*gsph[4] + -0.47308734787878*gsph[6] + 0.6258357354491761*gsph[8];
        gcart[    cart_stride] = 2.5033429417967046*gsph[0] + -0.94617469575756*gsph[2];
        gcart[2 * cart_stride] = -2.0071396306718676*gsph[5] + 1.7701307697799304*gsph[7];
        gcart[3 * cart_stride] = 0.6347132814912259*gsph[4] + -3.755014412695057*gsph[8];
        gcart[4 * cart_stride] = 5.310392309339791*gsph[1] + -2.0071396306718676*gsph[3];
        gcart[5 * cart_stride] = -2.5388531259649034*gsph[4] + 2.8385240872726802*gsph[6];
        gcart[6 * cart_stride] = -2.5033429417967046*gsph[0] + -0.94617469575756*gsph[2];
        gcart[7 * cart_stride] = -2.0071396306718676*gsph[5] + -5.310392309339791*gsph[7];
        gcart[8 * cart_stride] = 5.6770481745453605*gsph[2];
        gcart[9 * cart_stride] = 2.676186174229157*gsph[5];
        gcart[10 * cart_stride] = 0.31735664074561293*gsph[4] + 0.47308734787878*gsph[6] + 0.6258357354491761*gsph[8];
        gcart[11 * cart_stride] = -1.7701307697799304*gsph[1] + -2.0071396306718676*gsph[3];
        gcart[12 * cart_stride] = -2.5388531259649034*gsph[4] + -2.8385240872726802*gsph[6];
        gcart[13 * cart_stride] = 2.676186174229157*gsph[3];
        gcart[14 * cart_stride] = 0.8462843753216345*gsph[4];
    } else if constexpr (L == 5) {
        gcart[              0] = 0.45294665119569694*gsph[6] + -0.4892382994352504*gsph[8] + 0.6563820568401701*gsph[10];
        gcart[    cart_stride] = 3.2819102842008507*gsph[0] + -1.467714898305751*gsph[2] + 0.45294665119569694*gsph[4];
        gcart[2 * cart_stride] = 1.754254836801354*gsph[5] + -2.396768392486662*gsph[7] + 2.075662314881041*gsph[9];
        gcart[3 * cart_stride] = 0.9058933023913939*gsph[6] + 0.9784765988705008*gsph[8] + -6.563820568401701*gsph[10];
        gcart[4 * cart_stride] = 8.302649259524165*gsph[1] + -4.793536784973324*gsph[3];
        gcart[5 * cart_stride] = -5.435359814348363*gsph[6] + 3.913906395482003*gsph[8];
        gcart[6 * cart_stride] = -6.563820568401701*gsph[0] + -0.9784765988705008*gsph[2] + 0.9058933023913939*gsph[4];
        gcart[7 * cart_stride] = 3.508509673602708*gsph[5] + -12.453973889286248*gsph[9];
        gcart[8 * cart_stride] = 11.741719186446009*gsph[2] + -5.435359814348363*gsph[4];
        gcart[9 * cart_stride] = -4.678012898136944*gsph[5] + 4.793536784973324*gsph[7];
        gcart[10 * cart_stride] = 0.45294665119569694*gsph[6] + 1.467714898305751*gsph[8] + 3.2819102842008507*gsph[10];
        gcart[11 * cart_stride] = -8.302649259524165*gsph[1] + -4.793536784973324*gsph[3];
        gcart[12 * cart_stride] = -5.435359814348363*gsph[6] + -11.741719186446009*gsph[8];
        gcart[13 * cart_stride] = 9.587073569946648*gsph[3];
        gcart[14 * cart_stride] = 3.6235732095655755*gsph[6];
        gcart[15 * cart_stride] = 0.6563820568401701*gsph[0] + 0.4892382994352504*gsph[2] + 0.45294665119569694*gsph[4];
        gcart[16 * cart_stride] = 1.754254836801354*gsph[5] + 2.396768392486662*gsph[7] + 2.075662314881041*gsph[9];
        gcart[17 * cart_stride] = -3.913906395482003*gsph[2] + -5.435359814348363*gsph[4];
        gcart[18 * cart_stride] = -4.678012898136944*gsph[5] + -4.793536784973324*gsph[7];
        gcart[19 * cart_stride] = 3.6235732095655755*gsph[4];
        gcart[20 * cart_stride] = 0.9356025796273888*gsph[5];
    } else if constexpr (L == 6) {
        gcart[              0] = -0.3178460113381421*gsph[6] + 0.4606026297574618*gsph[8] + -0.5045649007287241*gsph[10] + 0.6831841051919144*gsph[12];
        gcart[    cart_stride] = 4.099104631151486*gsph[0] + -2.0182596029148963*gsph[2] + 0.9212052595149236*gsph[4];
        gcart[2 * cart_stride] = 2.913106812593657*gsph[7] + -2.7636157785447706*gsph[9] + 2.3666191622317525*gsph[11];
        gcart[3 * cart_stride] = -0.9535380340144264*gsph[6] + 0.4606026297574618*gsph[8] + 2.52282450364362*gsph[10] + -10.247761577878716*gsph[12];
        gcart[4 * cart_stride] = 11.833095811158763*gsph[1] + -8.29084733563431*gsph[3] + 2.913106812593657*gsph[5];
        gcart[5 * cart_stride] = 5.721228204086558*gsph[6] + -7.369642076119389*gsph[8] + 5.045649007287242*gsph[10];
        gcart[6 * cart_stride] = -13.663682103838289*gsph[0] + 1.8424105190298472*gsph[4];
        gcart[7 * cart_stride] = 5.826213625187314*gsph[7] + 5.527231557089541*gsph[9] + -23.666191622317527*gsph[11];
        gcart[8 * cart_stride] = 20.182596029148968*gsph[2] + -14.739284152238778*gsph[4];
        gcart[9 * cart_stride] = -11.652427250374627*gsph[7] + 7.369642076119389*gsph[9];
        gcart[10 * cart_stride] = -0.9535380340144264*gsph[6] + -0.4606026297574618*gsph[8] + 2.52282450364362*gsph[10] + 10.247761577878716*gsph[12];
        gcart[11 * cart_stride] = -23.666191622317527*gsph[1] + -5.527231557089541*gsph[3] + 5.826213625187314*gsph[5];
        gcart[12 * cart_stride] = 11.442456408173117*gsph[6] + -30.273894043723452*gsph[10];
        gcart[13 * cart_stride] = 22.108926228358165*gsph[3] + -11.652427250374627*gsph[5];
        gcart[14 * cart_stride] = -7.628304272115411*gsph[6] + 7.369642076119389*gsph[8];
        gcart[15 * cart_stride] = 4.099104631151486*gsph[0] + 2.0182596029148963*gsph[2] + 0.9212052595149236*gsph[4];
        gcart[16 * cart_stride] = 2.913106812593657*gsph[7] + 8.29084733563431*gsph[9] + 11.833095811158763*gsph[11];
        gcart[17 * cart_stride] = -20.182596029148968*gsph[2] + -14.739284152238778*gsph[4];
        gcart[18 * cart_stride] = -11.652427250374627*gsph[7] + -22.108926228358165*gsph[9];
        gcart[19 * cart_stride] = 14.739284152238778*gsph[4];
        gcart[20 * cart_stride] = 4.6609709001498505*gsph[7];
        gcart[21 * cart_stride] = -0.3178460113381421*gsph[6] + -0.4606026297574618*gsph[8] + -0.5045649007287241*gsph[10] + -0.6831841051919144*gsph[12];
        gcart[22 * cart_stride] = 2.3666191622317525*gsph[1] + 2.7636157785447706*gsph[3] + 2.913106812593657*gsph[5];
        gcart[23 * cart_stride] = 5.721228204086558*gsph[6] + 7.369642076119389*gsph[8] + 5.045649007287242*gsph[10];
        gcart[24 * cart_stride] = -7.369642076119389*gsph[3] + -11.652427250374627*gsph[5];
        gcart[25 * cart_stride] = -7.628304272115411*gsph[6] + -7.369642076119389*gsph[8];
        gcart[26 * cart_stride] = 4.6609709001498505*gsph[5];
        gcart[27 * cart_stride] = 1.0171072362820548*gsph[6];
    } else if constexpr (L == 7) {
        gcart[              0] = -0.4516580379125866*gsph[8] + 0.4693768015868821*gsph[10] + -0.5189155787202604*gsph[12] + 0.7071627325245963*gsph[14];
        gcart[    cart_stride] = 4.950139127672174*gsph[0] + -2.594577893601302*gsph[2] + 1.4081304047606462*gsph[4] + -0.4516580379125866*gsph[6];
        gcart[2 * cart_stride] = -2.389949691920173*gsph[7] + 3.3189951933373707*gsph[9] + -3.1134934723215624*gsph[11] + 2.6459606618019*gsph[13];
        gcart[3 * cart_stride] = -1.35497411373776*gsph[8] + -0.4693768015868821*gsph[10] + 4.670240208482344*gsph[12] + -14.850417383016522*gsph[14];
        gcart[4 * cart_stride] = 15.8757639708114*gsph[1] + -12.45397388928625*gsph[3] + 6.637990386674741*gsph[5];
        gcart[5 * cart_stride] = 10.839792909902078*gsph[8] + -9.38753603173764*gsph[10] + 6.226986944643125*gsph[12];
        gcart[6 * cart_stride] = -24.75069563836087*gsph[0] + 2.594577893601302*gsph[2] + 2.3468840079344107*gsph[4] + -1.35497411373776*gsph[6];
        gcart[7 * cart_stride] = -7.169849075760519*gsph[7] + 3.3189951933373707*gsph[9] + 15.567467361607811*gsph[11] + -39.6894099270285*gsph[13];
        gcart[8 * cart_stride] = 31.134934723215622*gsph[2] + -28.162608095212924*gsph[4] + 10.839792909902078*gsph[6];
        gcart[9 * cart_stride] = 14.339698151521036*gsph[7] + -17.701307697799308*gsph[9] + 10.378311574405208*gsph[11];
        gcart[10 * cart_stride] = -1.35497411373776*gsph[8] + -2.3468840079344107*gsph[10] + 2.594577893601302*gsph[12] + 24.75069563836087*gsph[14];
        gcart[11 * cart_stride] = -52.919213236038004*gsph[1] + 13.275980773349483*gsph[5];
        gcart[12 * cart_stride] = 21.679585819804156*gsph[8] + 18.77507206347528*gsph[10] + -62.269869446431244*gsph[12];
        gcart[13 * cart_stride] = 41.51324629762083*gsph[3] + -35.402615395598616*gsph[5];
        gcart[14 * cart_stride] = -21.679585819804156*gsph[8] + 12.516714708983523*gsph[10];
        gcart[15 * cart_stride] = 14.850417383016522*gsph[0] + 4.670240208482344*gsph[2] + 0.4693768015868821*gsph[4] + -1.35497411373776*gsph[6];
        gcart[16 * cart_stride] = -7.169849075760519*gsph[7] + -3.3189951933373707*gsph[9] + 15.567467361607811*gsph[11] + 39.6894099270285*gsph[13];
        gcart[17 * cart_stride] = -62.269869446431244*gsph[2] + -18.77507206347528*gsph[4] + 21.679585819804156*gsph[6];
        gcart[18 * cart_stride] = 28.679396303042072*gsph[7] + -62.269869446431244*gsph[11];
        gcart[19 * cart_stride] = 37.55014412695057*gsph[4] + -21.679585819804156*gsph[6];
        gcart[20 * cart_stride] = -11.47175852121683*gsph[7] + 10.620784618679586*gsph[9];
        gcart[21 * cart_stride] = -0.4516580379125866*gsph[8] + -1.4081304047606462*gsph[10] + -2.594577893601302*gsph[12] + -4.950139127672174*gsph[14];
        gcart[22 * cart_stride] = 15.8757639708114*gsph[1] + 12.45397388928625*gsph[3] + 6.637990386674741*gsph[5];
        gcart[23 * cart_stride] = 10.839792909902078*gsph[8] + 28.162608095212924*gsph[10] + 31.134934723215622*gsph[12];
        gcart[24 * cart_stride] = -41.51324629762083*gsph[3] + -35.402615395598616*gsph[5];
        gcart[25 * cart_stride] = -21.679585819804156*gsph[8] + -37.55014412695057*gsph[10];
        gcart[26 * cart_stride] = 21.241569237359172*gsph[5];
        gcart[27 * cart_stride] = 5.781222885281109*gsph[8];
        gcart[28 * cart_stride] = -0.7071627325245963*gsph[0] + -0.5189155787202604*gsph[2] + -0.4693768015868821*gsph[4] + -0.4516580379125866*gsph[6];
        gcart[29 * cart_stride] = -2.389949691920173*gsph[7] + -3.3189951933373707*gsph[9] + -3.1134934723215624*gsph[11] + -2.6459606618019*gsph[13];
        gcart[30 * cart_stride] = 6.226986944643125*gsph[2] + 9.38753603173764*gsph[4] + 10.839792909902078*gsph[6];
        gcart[31 * cart_stride] = 14.339698151521036*gsph[7] + 17.701307697799308*gsph[9] + 10.378311574405208*gsph[11];
        gcart[32 * cart_stride] = -12.516714708983523*gsph[4] + -21.679585819804156*gsph[6];
        gcart[33 * cart_stride] = -11.47175852121683*gsph[7] + -10.620784618679586*gsph[9];
        gcart[34 * cart_stride] = 5.781222885281109*gsph[6];
        gcart[35 * cart_stride] = 1.092548430592079*gsph[7];
    } else if constexpr (L == 8) {
        gcart[              0] = 0.3180369672047749*gsph[8] + -0.4561522584349095*gsph[10] + 0.4784165247593308*gsph[12] + -0.5323327660595425*gsph[14] + 0.72892666017483*gsph[16];
        gcart[    cart_stride] = 5.83141328139864*gsph[0] + -3.193996596357255*gsph[2] + 1.913666099037323*gsph[4] + -0.912304516869819*gsph[6];
        gcart[2 * cart_stride] = -3.8164436064573*gsph[9] + 3.705798465886632*gsph[11] + -3.449910622098108*gsph[13] + 2.91570664069932*gsph[15];
        gcart[3 * cart_stride] = 1.272147868819099*gsph[8] + -0.912304516869819*gsph[10] + -1.913666099037323*gsph[12] + 7.452658724833595*gsph[14] + -20.40994648489524*gsph[16];
        gcart[4 * cart_stride] = 20.40994648489524*gsph[1] + -17.24955311049054*gsph[3] + 11.1173953976599*gsph[5] + -3.8164436064573*gsph[7];
        gcart[5 * cart_stride] = -10.1771829505528*gsph[8] + 13.68456775304729*gsph[10] + -11.48199659422394*gsph[12] + 7.452658724833595*gsph[14];
        gcart[6 * cart_stride] = -40.81989296979048*gsph[0] + 7.452658724833595*gsph[2] + 1.913666099037323*gsph[4] + -2.736913550609457*gsph[6];
        gcart[7 * cart_stride] = -11.4493308193719*gsph[9] + -3.705798465886632*gsph[11] + 31.04919559888297*gsph[13] + -61.22983945468572*gsph[15];
        gcart[8 * cart_stride] = 44.71595234900157*gsph[2] + -45.92798637689575*gsph[4] + 27.36913550609457*gsph[6];
        gcart[9 * cart_stride] = 30.5315488516584*gsph[9] + -24.70532310591088*gsph[11] + 13.79964248839243*gsph[13];
        gcart[10 * cart_stride] = 1.908221803228649*gsph[8] + -4.784165247593307*gsph[12] + 51.0248662122381*gsph[16];
        gcart[11 * cart_stride] = -102.0497324244762*gsph[1] + 17.24955311049054*gsph[3] + 18.52899232943316*gsph[5] + -11.4493308193719*gsph[7];
        gcart[12 * cart_stride] = -30.53154885165839*gsph[8] + 13.68456775304729*gsph[10] + 57.40998297111968*gsph[12] + -111.7898808725039*gsph[14];
        gcart[13 * cart_stride] = 68.99821244196217*gsph[3] + -74.11596931773265*gsph[5] + 30.5315488516584*gsph[7];
        gcart[14 * cart_stride] = 30.53154885165839*gsph[8] + -36.49218067479276*gsph[10] + 19.13666099037323*gsph[12];
        gcart[15 * cart_stride] = 40.81989296979048*gsph[0] + 7.452658724833595*gsph[2] + -1.913666099037323*gsph[4] + -2.736913550609457*gsph[6];
        gcart[16 * cart_stride] = -11.4493308193719*gsph[9] + -18.52899232943316*gsph[11] + 17.24955311049054*gsph[13] + 102.0497324244762*gsph[15];
        gcart[17 * cart_stride] = -149.0531744966719*gsph[2] + 54.73827101218914*gsph[6];
        gcart[18 * cart_stride] = 61.06309770331679*gsph[9] + 49.41064621182176*gsph[11] + -137.9964248839243*gsph[13];
        gcart[19 * cart_stride] = 76.54664396149292*gsph[4] + -72.98436134958553*gsph[6];
        gcart[20 * cart_stride] = -36.63785862199007*gsph[9] + 19.7642584847287*gsph[11];
        gcart[21 * cart_stride] = 1.272147868819099*gsph[8] + 0.912304516869819*gsph[10] + -1.913666099037323*gsph[12] + -7.452658724833595*gsph[14] + -20.40994648489524*gsph[16];
        gcart[22 * cart_stride] = 61.22983945468572*gsph[1] + 31.04919559888297*gsph[3] + 3.705798465886632*gsph[5] + -11.4493308193719*gsph[7];
        gcart[23 * cart_stride] = -30.53154885165839*gsph[8] + -13.68456775304729*gsph[10] + 57.40998297111968*gsph[12] + 111.7898808725039*gsph[14];
        gcart[24 * cart_stride] = -137.9964248839243*gsph[3] + -49.41064621182176*gsph[5] + 61.06309770331679*gsph[7];
        gcart[25 * cart_stride] = 61.06309770331677*gsph[8] + -114.8199659422394*gsph[12];
        gcart[26 * cart_stride] = 59.29277545418611*gsph[5] + -36.63785862199007*gsph[7];
        gcart[27 * cart_stride] = -16.28349272088447*gsph[8] + 14.5968722699171*gsph[10];
        gcart[28 * cart_stride] = -5.83141328139864*gsph[0] + -3.193996596357255*gsph[2] + -1.913666099037323*gsph[4] + -0.912304516869819*gsph[6];
        gcart[29 * cart_stride] = -3.8164436064573*gsph[9] + -11.1173953976599*gsph[11] + -17.24955311049054*gsph[13] + -20.40994648489524*gsph[15];
        gcart[30 * cart_stride] = 44.71595234900157*gsph[2] + 45.92798637689575*gsph[4] + 27.36913550609457*gsph[6];
        gcart[31 * cart_stride] = 30.5315488516584*gsph[9] + 74.11596931773265*gsph[11] + 68.99821244196217*gsph[13];
        gcart[32 * cart_stride] = -76.54664396149292*gsph[4] + -72.98436134958553*gsph[6];
        gcart[33 * cart_stride] = -36.63785862199007*gsph[9] + -59.29277545418611*gsph[11];
        gcart[34 * cart_stride] = 29.19374453983421*gsph[6];
        gcart[35 * cart_stride] = 6.978639737521918*gsph[9];
        gcart[36 * cart_stride] = 0.3180369672047749*gsph[8] + 0.4561522584349095*gsph[10] + 0.4784165247593308*gsph[12] + 0.5323327660595425*gsph[14] + 0.72892666017483*gsph[16];
        gcart[37 * cart_stride] = -2.91570664069932*gsph[1] + -3.449910622098108*gsph[3] + -3.705798465886632*gsph[5] + -3.8164436064573*gsph[7];
        gcart[38 * cart_stride] = -10.1771829505528*gsph[8] + -13.68456775304729*gsph[10] + -11.48199659422394*gsph[12] + -7.452658724833595*gsph[14];
        gcart[39 * cart_stride] = 13.79964248839243*gsph[3] + 24.70532310591088*gsph[5] + 30.5315488516584*gsph[7];
        gcart[40 * cart_stride] = 30.53154885165839*gsph[8] + 36.49218067479276*gsph[10] + 19.13666099037323*gsph[12];
        gcart[41 * cart_stride] = -19.7642584847287*gsph[5] + -36.63785862199007*gsph[7];
        gcart[42 * cart_stride] = -16.28349272088447*gsph[8] + -14.5968722699171*gsph[10];
        gcart[43 * cart_stride] = 6.978639737521918*gsph[7];
        gcart[44 * cart_stride] = 1.16310662292032*gsph[8];
    } else if constexpr (L == 9) {
        gcart[              0] = 0.451093112065591*gsph[10] + -0.4617085200161945*gsph[12] + 0.4873782790390186*gsph[14] + -0.5449054813440533*gsph[16] + 0.7489009518531882*gsph[18];
        gcart[    cart_stride] = 6.740108566678694*gsph[0] + -3.814338369408373*gsph[2] + 2.436891395195093*gsph[4] + -1.385125560048583*gsph[6] + 0.451093112065591*gsph[8];
        gcart[2 * cart_stride] = 3.026024588281776*gsph[9] + -4.23162848396049*gsph[11] + 4.077699238729173*gsph[13] + -3.775215916042701*gsph[15] + 3.177317648954698*gsph[17];
        gcart[3 * cart_stride] = 1.804372448262364*gsph[10] + -3.899026232312149*gsph[14] + 10.89810962688107*gsph[16] + -26.96043426671477*gsph[18];
        gcart[4 * cart_stride] = 25.41854119163758*gsph[1] + -22.65129549625621*gsph[3] + 16.31079695491669*gsph[5] + -8.46325696792098*gsph[7];
        gcart[5 * cart_stride] = -18.04372448262364*gsph[10] + 16.621506720583*gsph[12] + -13.64659181309252*gsph[14] + 8.718487701504852*gsph[16];
        gcart[6 * cart_stride] = -62.9076799556678*gsph[0] + 15.25735347763349*gsph[2] + -3.693668160129556*gsph[6] + 1.804372448262364*gsph[8];
        gcart[7 * cart_stride] = 12.1040983531271*gsph[9] + -8.46325696792098*gsph[11] + -16.31079695491669*gsph[13] + 52.85302282459782*gsph[15] + -88.96489417073154*gsph[17];
        gcart[8 * cart_stride] = 61.02941391053396*gsph[2] + -68.23295906546261*gsph[4] + 49.864520161749*gsph[6] + -18.04372448262364*gsph[8];
        gcart[9 * cart_stride] = -32.27759560833895*gsph[9] + 42.3162848396049*gsph[11] + -32.62159390983339*gsph[13] + 17.61767427486594*gsph[15];
        gcart[10 * cart_stride] = 2.706558672393546*gsph[10] + 2.770251120097167*gsph[12] + -6.82329590654626*gsph[14] + -7.628676738816745*gsph[16] + 94.36151993350171*gsph[18];
        gcart[11 * cart_stride] = -177.9297883414631*gsph[1] + 52.85302282459782*gsph[3] + 16.31079695491669*gsph[5] + -25.38977090376294*gsph[7];
        gcart[12 * cart_stride] = -54.13117344787092*gsph[10] + -16.621506720583*gsph[12] + 122.8193263178327*gsph[14] + -183.0882417316019*gsph[16];
        gcart[13 * cart_stride] = 105.7060456491956*gsph[3] + -130.4863756393335*gsph[5] + 84.6325696792098*gsph[7];
        gcart[14 * cart_stride] = 72.17489793049457*gsph[10] + -55.40502240194333*gsph[12] + 27.29318362618504*gsph[14];
        gcart[15 * cart_stride] = 94.36151993350171*gsph[0] + 7.628676738816745*gsph[2] + -6.82329590654626*gsph[4] + -2.770251120097167*gsph[6] + 2.706558672393546*gsph[8];
        gcart[16 * cart_stride] = 18.15614752969066*gsph[9] + -40.77699238729173*gsph[13] + 222.4122354268289*gsph[17];
        gcart[17 * cart_stride] = -305.1470695526698*gsph[2] + 68.23295906546261*gsph[4] + 83.107533602915*gsph[6] + -54.13117344787092*gsph[8];
        gcart[18 * cart_stride] = -96.83278682501685*gsph[9] + 42.3162848396049*gsph[11] + 163.1079695491669*gsph[13] + -264.2651141229891*gsph[15];
        gcart[19 * cart_stride] = 136.4659181309252*gsph[4] + -166.21506720583*gsph[6] + 72.17489793049457*gsph[8];
        gcart[20 * cart_stride] = 58.0996720950101*gsph[9] + -67.70605574336784*gsph[11] + 32.62159390983339*gsph[13];
        gcart[21 * cart_stride] = 1.804372448262364*gsph[10] + 3.693668160129556*gsph[12] + -15.25735347763349*gsph[16] + -62.9076799556678*gsph[18];
        gcart[22 * cart_stride] = 177.9297883414631*gsph[1] + 52.85302282459782*gsph[3] + -16.31079695491669*gsph[5] + -25.38977090376294*gsph[7];
        gcart[23 * cart_stride] = -54.13117344787092*gsph[10] + -83.107533602915*gsph[12] + 68.23295906546261*gsph[14] + 305.1470695526698*gsph[16];
        gcart[24 * cart_stride] = -352.3534854973187*gsph[3] + 169.2651393584196*gsph[7];
        gcart[25 * cart_stride] = 144.3497958609891*gsph[10] + 110.8100448038867*gsph[12] + -272.9318362618504*gsph[14];
        gcart[26 * cart_stride] = 130.4863756393335*gsph[5] + -135.4121114867357*gsph[7];
        gcart[27 * cart_stride] = -57.73991834439565*gsph[10] + 29.54934528103645*gsph[12];
        gcart[28 * cart_stride] = -26.96043426671477*gsph[0] + -10.89810962688107*gsph[2] + -3.899026232312149*gsph[4] + 1.804372448262364*gsph[8];
        gcart[29 * cart_stride] = 12.1040983531271*gsph[9] + 8.46325696792098*gsph[11] + -16.31079695491669*gsph[13] + -52.85302282459782*gsph[15] + -88.96489417073154*gsph[17];
        gcart[30 * cart_stride] = 183.0882417316019*gsph[2] + 122.8193263178327*gsph[4] + 16.621506720583*gsph[6] + -54.13117344787092*gsph[8];
        gcart[31 * cart_stride] = -96.83278682501685*gsph[9] + -42.3162848396049*gsph[11] + 163.1079695491669*gsph[13] + 264.2651141229891*gsph[15];
        gcart[32 * cart_stride] = -272.9318362618504*gsph[4] + -110.8100448038867*gsph[6] + 144.3497958609891*gsph[8];
        gcart[33 * cart_stride] = 116.1993441900202*gsph[9] + -195.7295634590003*gsph[13];
        gcart[34 * cart_stride] = 88.64803584310934*gsph[6] + -57.73991834439565*gsph[8];
        gcart[35 * cart_stride] = -22.1332084171467*gsph[9] + 19.34458735524795*gsph[11];
        gcart[36 * cart_stride] = 0.451093112065591*gsph[10] + 1.385125560048583*gsph[12] + 2.436891395195093*gsph[14] + 3.814338369408373*gsph[16] + 6.740108566678694*gsph[18];
        gcart[37 * cart_stride] = -25.41854119163758*gsph[1] + -22.65129549625621*gsph[3] + -16.31079695491669*gsph[5] + -8.46325696792098*gsph[7];
        gcart[38 * cart_stride] = -18.04372448262364*gsph[10] + -49.864520161749*gsph[12] + -68.23295906546261*gsph[14] + -61.02941391053396*gsph[16];
        gcart[39 * cart_stride] = 105.7060456491956*gsph[3] + 130.4863756393335*gsph[5] + 84.6325696792098*gsph[7];
        gcart[40 * cart_stride] = 72.17489793049457*gsph[10] + 166.21506720583*gsph[12] + 136.4659181309252*gsph[14];
        gcart[41 * cart_stride] = -130.4863756393335*gsph[5] + -135.4121114867357*gsph[7];
        gcart[42 * cart_stride] = -57.73991834439565*gsph[10] + -88.64803584310934*gsph[12];
        gcart[43 * cart_stride] = 38.68917471049591*gsph[7];
        gcart[44 * cart_stride] = 8.248559763485094*gsph[10];
        gcart[45 * cart_stride] = 0.7489009518531882*gsph[0] + 0.5449054813440533*gsph[2] + 0.4873782790390186*gsph[4] + 0.4617085200161945*gsph[6] + 0.451093112065591*gsph[8];
        gcart[46 * cart_stride] = 3.026024588281776*gsph[9] + 4.23162848396049*gsph[11] + 4.077699238729173*gsph[13] + 3.775215916042701*gsph[15] + 3.177317648954698*gsph[17];
        gcart[47 * cart_stride] = -8.718487701504852*gsph[2] + -13.64659181309252*gsph[4] + -16.621506720583*gsph[6] + -18.04372448262364*gsph[8];
        gcart[48 * cart_stride] = -32.27759560833895*gsph[9] + -42.3162848396049*gsph[11] + -32.62159390983339*gsph[13] + -17.61767427486594*gsph[15];
        gcart[49 * cart_stride] = 27.29318362618504*gsph[4] + 55.40502240194333*gsph[6] + 72.17489793049457*gsph[8];
        gcart[50 * cart_stride] = 58.0996720950101*gsph[9] + 67.70605574336784*gsph[11] + 32.62159390983339*gsph[13];
        gcart[51 * cart_stride] = -29.54934528103645*gsph[6] + -57.73991834439565*gsph[8];
        gcart[52 * cart_stride] = -22.1332084171467*gsph[9] + -19.34458735524795*gsph[11];
        gcart[53 * cart_stride] = 8.248559763485094*gsph[8];
        gcart[54 * cart_stride] = 1.229622689841484*gsph[9];
    } else if constexpr (L == 10) {
        gcart[              0] = -0.3181304937373671*gsph[10] + 0.4540511313802278*gsph[12] + -0.4677441816782422*gsph[14] + 0.4961176240878564*gsph[16] + -0.5567269327204184*gsph[18] + 0.7673951182219901*gsph[20];
        gcart[    cart_stride] = 7.673951182219901*gsph[0] + -4.453815461763347*gsph[2] + 2.976705744527138*gsph[4] + -1.870976726712969*gsph[6] + 0.9081022627604556*gsph[8];
        gcart[2 * cart_stride] = 4.718637772708116*gsph[11] + -4.630431158153326*gsph[13] + 4.437410929184535*gsph[15] + -4.091090733689417*gsph[17] + 3.431895299891715*gsph[19];
        gcart[3 * cart_stride] = -1.590652468686835*gsph[10] + 1.362153394140683*gsph[12] + 1.403232545034726*gsph[14] + -6.449529113142133*gsph[16] + 15.0316271834513*gsph[18] + -34.53278031998956*gsph[20];
        gcart[4 * cart_stride] = 30.88705769902543*gsph[1] + -28.63763513582592*gsph[3] + 22.18705464592268*gsph[5] + -13.89129347445998*gsph[7] + 4.718637772708116*gsph[9];
        gcart[5 * cart_stride] = 15.90652468686835*gsph[10] + -21.79445430625093*gsph[12] + 19.64525563048617*gsph[14] + -15.8757639708114*gsph[16] + 10.02108478896753*gsph[18];
        gcart[6 * cart_stride] = -92.08741418663881*gsph[0] + 26.72289277058008*gsph[2] + -3.968940992702851*gsph[4] + -3.741953453425937*gsph[6] + 3.632409051041822*gsph[8];
        gcart[7 * cart_stride] = 18.87455109083247*gsph[11] + -35.49928743347628*gsph[15] + 81.82181467378834*gsph[17] + -123.5482307961017*gsph[19];
        gcart[8 * cart_stride] = 80.16867831174027*gsph[2] + -95.25458382486842*gsph[4] + 78.58102252194469*gsph[6] + -43.58890861250187*gsph[8];
        gcart[9 * cart_stride] = -62.91517030277488*gsph[11] + 55.56517389783991*gsph[13] + -41.41583533905566*gsph[15] + 21.81915057967689*gsph[17];
        gcart[10 * cart_stride] = -3.181304937373671*gsph[10] + 0.9081022627604556*gsph[12] + 6.548418543495391*gsph[14] + -6.945646737229989*gsph[16] + -23.38253117425757*gsph[18] + 161.1529748266179*gsph[20];
        gcart[11 * cart_stride] = -288.2792051909041*gsph[1] + 114.5505405433037*gsph[3] + -37.04344926522661*gsph[7] + 18.87455109083247*gsph[9];
        gcart[12 * cart_stride] = 63.62609874747341*gsph[10] + -43.58890861250187*gsph[12] + -78.58102252194469*gsph[14] + 222.2606955913596*gsph[16] + -280.590374091091*gsph[18];
        gcart[13 * cart_stride] = 152.7340540577382*gsph[3] + -207.0791766952783*gsph[5] + 166.6955216935197*gsph[7] + -62.91517030277488*gsph[9];
        gcart[14 * cart_stride] = -84.83479832996456*gsph[10] + 108.9722715312547*gsph[12] + -78.58102252194469*gsph[14] + 37.04344926522661*gsph[16];
        gcart[15 * cart_stride] = 193.3835697919415*gsph[0] + -13.89129347445998*gsph[4] + 5.448613576562733*gsph[8];
        gcart[16 * cart_stride] = 28.3118266362487*gsph[11] + 27.78258694891996*gsph[13] + -62.12375300858349*gsph[15] + -57.27527027165184*gsph[17] + 432.4188077863561*gsph[19];
        gcart[17 * cart_stride] = -561.1807481821819*gsph[2] + 222.2606955913596*gsph[4] + 78.58102252194469*gsph[6] + -130.7667258375056*gsph[8];
        gcart[18 * cart_stride] = -188.7455109083247*gsph[11] + -55.56517389783991*gsph[13] + 372.742518051501*gsph[15] + -458.2021621732147*gsph[17];
        gcart[19 * cart_stride] = 222.2606955913597*gsph[4] + -314.3240900877788*gsph[6] + 217.9445430625093*gsph[8];
        gcart[20 * cart_stride] = 150.9964087266597*gsph[11] + -111.1303477956798*gsph[13] + 49.6990024068668*gsph[15];
        gcart[21 * cart_stride] = -3.181304937373671*gsph[10] + -0.9081022627604556*gsph[12] + 6.548418543495391*gsph[14] + 6.945646737229989*gsph[16] + -23.38253117425757*gsph[18] + -161.1529748266179*gsph[20];
        gcart[22 * cart_stride] = 432.4188077863561*gsph[1] + 57.27527027165184*gsph[3] + -62.12375300858349*gsph[5] + -27.78258694891996*gsph[7] + 28.3118266362487*gsph[9];
        gcart[23 * cart_stride] = 95.43914812121012*gsph[10] + -196.4525563048617*gsph[14] + 701.4759352277273*gsph[18];
        gcart[24 * cart_stride] = -763.6702702886912*gsph[3] + 207.0791766952783*gsph[5] + 277.8258694891996*gsph[7] + -188.7455109083247*gsph[9];
        gcart[25 * cart_stride] = -254.5043949898937*gsph[10] + 108.9722715312547*gsph[12] + 392.9051126097235*gsph[14] + -555.6517389783992*gsph[16];
        gcart[26 * cart_stride] = 248.495012034334*gsph[5] + -333.3910433870395*gsph[7] + 150.9964087266597*gsph[9];
        gcart[27 * cart_stride] = 101.8017579959575*gsph[10] + -116.2370896333383*gsph[12] + 52.38734834796313*gsph[14];
        gcart[28 * cart_stride] = -92.08741418663881*gsph[0] + -26.72289277058008*gsph[2] + -3.968940992702851*gsph[4] + 3.741953453425937*gsph[6] + 3.632409051041822*gsph[8];
        gcart[29 * cart_stride] = 18.87455109083247*gsph[11] + 37.04344926522661*gsph[13] + -114.5505405433037*gsph[17] + -288.2792051909041*gsph[19];
        gcart[30 * cart_stride] = 561.1807481821819*gsph[2] + 222.2606955913596*gsph[4] + -78.58102252194469*gsph[6] + -130.7667258375056*gsph[8];
        gcart[31 * cart_stride] = -188.7455109083247*gsph[11] + -277.8258694891996*gsph[13] + 207.0791766952783*gsph[15] + 763.6702702886912*gsph[17];
        gcart[32 * cart_stride] = -740.8689853045323*gsph[4] + 435.8890861250187*gsph[8];
        gcart[33 * cart_stride] = 301.9928174533194*gsph[11] + 222.2606955913596*gsph[13] + -496.990024068668*gsph[15];
        gcart[34 * cart_stride] = 209.5493933918525*gsph[6] + -232.4741792666766*gsph[8];
        gcart[35 * cart_stride] = -86.28366212951984*gsph[11] + 42.33537058883041*gsph[13];
        gcart[36 * cart_stride] = -1.590652468686835*gsph[10] + -1.362153394140683*gsph[12] + 1.403232545034726*gsph[14] + 6.449529113142133*gsph[16] + 15.0316271834513*gsph[18] + 34.53278031998956*gsph[20];
        gcart[37 * cart_stride] = -123.5482307961017*gsph[1] + -81.82181467378834*gsph[3] + -35.49928743347628*gsph[5] + 18.87455109083247*gsph[9];
        gcart[38 * cart_stride] = 63.62609874747341*gsph[10] + 43.58890861250187*gsph[12] + -78.58102252194469*gsph[14] + -222.2606955913596*gsph[16] + -280.590374091091*gsph[18];
        gcart[39 * cart_stride] = 458.2021621732147*gsph[3] + 372.742518051501*gsph[5] + 55.56517389783991*gsph[7] + -188.7455109083247*gsph[9];
        gcart[40 * cart_stride] = -254.5043949898937*gsph[10] + -108.9722715312547*gsph[12] + 392.9051126097235*gsph[14] + 555.6517389783992*gsph[16];
        gcart[41 * cart_stride] = -496.990024068668*gsph[5] + -222.2606955913596*gsph[7] + 301.9928174533194*gsph[9];
        gcart[42 * cart_stride] = 203.6035159919149*gsph[10] + -314.3240900877788*gsph[14];
        gcart[43 * cart_stride] = 127.0061117664912*gsph[7] + -86.28366212951984*gsph[9];
        gcart[44 * cart_stride] = -29.08621657027356*gsph[10] + 24.9079477785725*gsph[12];
        gcart[45 * cart_stride] = 7.673951182219901*gsph[0] + 4.453815461763347*gsph[2] + 2.976705744527138*gsph[4] + 1.870976726712969*gsph[6] + 0.9081022627604556*gsph[8];
        gcart[46 * cart_stride] = 4.718637772708116*gsph[11] + 13.89129347445998*gsph[13] + 22.18705464592268*gsph[15] + 28.63763513582592*gsph[17] + 30.88705769902543*gsph[19];
        gcart[47 * cart_stride] = -80.16867831174027*gsph[2] + -95.25458382486842*gsph[4] + -78.58102252194469*gsph[6] + -43.58890861250187*gsph[8];
        gcart[48 * cart_stride] = -62.91517030277488*gsph[11] + -166.6955216935197*gsph[13] + -207.0791766952783*gsph[15] + -152.7340540577382*gsph[17];
        gcart[49 * cart_stride] = 222.2606955913597*gsph[4] + 314.3240900877788*gsph[6] + 217.9445430625093*gsph[8];
        gcart[50 * cart_stride] = 150.9964087266597*gsph[11] + 333.3910433870395*gsph[13] + 248.495012034334*gsph[15];
        gcart[51 * cart_stride] = -209.5493933918525*gsph[6] + -232.4741792666766*gsph[8];
        gcart[52 * cart_stride] = -86.28366212951984*gsph[11] + -127.0061117664912*gsph[13];
        gcart[53 * cart_stride] = 49.815895557145*gsph[8];
        gcart[54 * cart_stride] = 9.587073569946648*gsph[11];
        gcart[55 * cart_stride] = -0.3181304937373671*gsph[10] + -0.4540511313802278*gsph[12] + -0.4677441816782422*gsph[14] + -0.4961176240878564*gsph[16] + -0.5567269327204184*gsph[18] + -0.7673951182219901*gsph[20];
        gcart[56 * cart_stride] = 3.431895299891715*gsph[1] + 4.091090733689417*gsph[3] + 4.437410929184535*gsph[5] + 4.630431158153326*gsph[7] + 4.718637772708116*gsph[9];
        gcart[57 * cart_stride] = 15.90652468686835*gsph[10] + 21.79445430625093*gsph[12] + 19.64525563048617*gsph[14] + 15.8757639708114*gsph[16] + 10.02108478896753*gsph[18];
        gcart[58 * cart_stride] = -21.81915057967689*gsph[3] + -41.41583533905566*gsph[5] + -55.56517389783991*gsph[7] + -62.91517030277488*gsph[9];
        gcart[59 * cart_stride] = -84.83479832996456*gsph[10] + -108.9722715312547*gsph[12] + -78.58102252194469*gsph[14] + -37.04344926522661*gsph[16];
        gcart[60 * cart_stride] = 49.6990024068668*gsph[5] + 111.1303477956798*gsph[7] + 150.9964087266597*gsph[9];
        gcart[61 * cart_stride] = 101.8017579959575*gsph[10] + 116.2370896333383*gsph[12] + 52.38734834796313*gsph[14];
        gcart[62 * cart_stride] = -42.33537058883041*gsph[7] + -86.28366212951984*gsph[9];
        gcart[63 * cart_stride] = -29.08621657027356*gsph[10] + -24.9079477785725*gsph[12];
        gcart[64 * cart_stride] = 9.587073569946648*gsph[9];
        gcart[65 * cart_stride] = 1.292720736456603*gsph[10];
    } else {
        gcart[              0] = NAN;
    }
}

template <int L>
__global__
static void left_cart2sph_inplace(double* cartesian_matrix, const int n_ao_cartesian, const int n_bas, const int i_bas_offset)
{
    constexpr int n_cartesian_of_l = (L + 1) * (L + 2) / 2;
    constexpr int n_spherical_of_l = 2 * L + 1;

    const int i_ao = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_bas = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_ao >= n_ao_cartesian || i_bas >= n_bas)
        return;

    double spherical_cache[n_spherical_of_l];
    cart2sph<L>(cartesian_matrix + (i_bas_offset + i_bas * n_cartesian_of_l) * n_ao_cartesian + i_ao, spherical_cache, n_ao_cartesian);

    for (int i = 0; i < n_spherical_of_l; i++)
        cartesian_matrix[(i_bas_offset + i_bas * n_cartesian_of_l + i) * n_ao_cartesian + i_ao] = spherical_cache[i];
}

template <int L>
__global__
static void left_sph2cart_inplace(double* cartesian_matrix, const int n_ao_cartesian, const int n_bas, const int i_bas_offset)
{
    constexpr int n_cartesian_of_l = (L + 1) * (L + 2) / 2;
    constexpr int n_spherical_of_l = 2 * L + 1;

    const int i_ao = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_bas = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_ao >= n_ao_cartesian || i_bas >= n_bas)
        return;

    double spherical_cache[n_spherical_of_l];
    for (int i = 0; i < n_spherical_of_l; i++)
        spherical_cache[i] = cartesian_matrix[(i_bas_offset + i_bas * n_cartesian_of_l + i) * n_ao_cartesian + i_ao];

    sph2cart<L>(cartesian_matrix + (i_bas_offset + i_bas * n_cartesian_of_l) * n_ao_cartesian + i_ao, spherical_cache, n_ao_cartesian);
}

template <int L>
__global__
static void left_sph2cart(double* cartesian_matrix, const double* spherical_matrix,
                          const int n_right, const int n_bas, const int i_cartesian_offset, const int i_spherical_offset,
                          const int* d_ao_idx)
{
    constexpr int n_cartesian_of_l = (L + 1) * (L + 2) / 2;
    constexpr int n_spherical_of_l = 2 * L + 1;

    const int i_ao = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_bas = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_ao >= n_right || i_bas >= n_bas)
        return;

    double spherical_cache[n_spherical_of_l];
    for (int i = 0; i < n_spherical_of_l; i++)
        spherical_cache[i] = spherical_matrix[d_ao_idx[i_spherical_offset + i_bas * n_spherical_of_l + i] * n_right + i_ao];

    sph2cart<L>(cartesian_matrix + (i_cartesian_offset + i_bas * n_cartesian_of_l) * n_right + i_ao, spherical_cache, n_right);
}

template <int L>
__global__
static void right_cart2sph_inplace(double* cartesian_matrix, const int n_ao_cartesian, const int n_bas, const int i_bas_offset)
{
    constexpr int n_cartesian_of_l = (L + 1) * (L + 2) / 2;
    constexpr int n_spherical_of_l = 2 * L + 1;

    const int i_ao = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_bas = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_ao >= n_ao_cartesian || i_bas >= n_bas)
        return;

    double spherical_cache[n_spherical_of_l];
    cart2sph<L>(cartesian_matrix + i_bas_offset + i_bas * n_cartesian_of_l + i_ao * n_ao_cartesian, spherical_cache, 1);

    for (int i = 0; i < n_spherical_of_l; i++)
        cartesian_matrix[i_bas_offset + i_bas * n_cartesian_of_l + i + i_ao * n_ao_cartesian] = spherical_cache[i];
}

template <int L>
__global__
static void right_sph2cart_inplace(double* cartesian_matrix, const int n_ao_cartesian, const int n_bas, const int i_bas_offset)
{
    constexpr int n_cartesian_of_l = (L + 1) * (L + 2) / 2;
    constexpr int n_spherical_of_l = 2 * L + 1;

    const int i_ao = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_bas = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_ao >= n_ao_cartesian || i_bas >= n_bas)
        return;

    double spherical_cache[n_spherical_of_l];
    for (int i = 0; i < n_spherical_of_l; i++)
        spherical_cache[i] = cartesian_matrix[i_bas_offset + i_bas * n_cartesian_of_l + i + i_ao * n_ao_cartesian];

    sph2cart<L>(cartesian_matrix + i_bas_offset + i_bas * n_cartesian_of_l + i_ao * n_ao_cartesian, spherical_cache, 1);
}

__global__
static void copy_spherical_cart2sph(const double* cartesian_matrix, double* spherical_matrix,
                                    const int n_ao_cartesian, const int n_ao_spherical,
                                    const int l_i, const int n_bas_i, const int cartesian_offset_i, const int spherical_offset_i,
                                    const int l_j, const int n_bas_j, const int cartesian_offset_j, const int spherical_offset_j,
                                    const int* d_ao_idx)
{
    const int i_bas = blockIdx.x * blockDim.x + threadIdx.x;
    const int j_bas = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_bas >= n_bas_i || j_bas >= n_bas_j)
        return;

    const int n_cartesian_of_l_i = (l_i + 1) * (l_i + 2) / 2;
    const int n_spherical_of_l_i = 2 * l_i + 1;
    const int n_cartesian_of_l_j = (l_j + 1) * (l_j + 2) / 2;
    const int n_spherical_of_l_j = 2 * l_j + 1;

    for (int i_spherical = 0; i_spherical < n_spherical_of_l_i; i_spherical++) {
        for (int j_spherical = 0; j_spherical < n_spherical_of_l_j; j_spherical++) {
            spherical_matrix[d_ao_idx[spherical_offset_i + i_bas * n_spherical_of_l_i + i_spherical] * n_ao_spherical
                           + d_ao_idx[spherical_offset_j + j_bas * n_spherical_of_l_j + j_spherical]]
            = cartesian_matrix[(cartesian_offset_i + i_bas * n_cartesian_of_l_i + i_spherical) * n_ao_cartesian
                             + (cartesian_offset_j + j_bas * n_cartesian_of_l_j + j_spherical)];
        }
    }
}

__global__
static void copy_spherical_sph2cart(double* cartesian_matrix, const double* spherical_matrix,
                                    const int n_ao_cartesian, const int n_ao_spherical,
                                    const int l_i, const int n_bas_i, const int cartesian_offset_i, const int spherical_offset_i,
                                    const int l_j, const int n_bas_j, const int cartesian_offset_j, const int spherical_offset_j,
                                    const int* d_ao_idx)
{
    const int i_bas = blockIdx.x * blockDim.x + threadIdx.x;
    const int j_bas = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_bas >= n_bas_i || j_bas >= n_bas_j)
        return;

    const int n_cartesian_of_l_i = (l_i + 1) * (l_i + 2) / 2;
    const int n_spherical_of_l_i = 2 * l_i + 1;
    const int n_cartesian_of_l_j = (l_j + 1) * (l_j + 2) / 2;
    const int n_spherical_of_l_j = 2 * l_j + 1;

    for (int i_spherical = 0; i_spherical < n_spherical_of_l_i; i_spherical++) {
        for (int j_spherical = 0; j_spherical < n_spherical_of_l_j; j_spherical++) {
            cartesian_matrix[(cartesian_offset_i + i_bas * n_cartesian_of_l_i + i_spherical) * n_ao_cartesian
                           + (cartesian_offset_j + j_bas * n_cartesian_of_l_j + j_spherical)]
            = spherical_matrix[d_ao_idx[spherical_offset_i + i_bas * n_spherical_of_l_i + i_spherical] * n_ao_spherical
                             + d_ao_idx[spherical_offset_j + j_bas * n_spherical_of_l_j + j_spherical]];
        }
    }
}

__global__
static void copy_cartesian_pad_to_unpad(const double* cartesian_matrix, double* spherical_matrix,
                                        const int n_ao_cartesian, const int n_ao_spherical,
                                        const int l_i, const int n_bas_i, const int i_pad_offset, const int i_unpad_offset,
                                        const int l_j, const int n_bas_j, const int j_pad_offset, const int j_unpad_offset,
                                        const int* d_ao_idx)
{
    const int i_bas = blockIdx.x * blockDim.x + threadIdx.x;
    const int j_bas = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_bas >= n_bas_i || j_bas >= n_bas_j)
        return;

    const int n_cartesian_of_l_i = (l_i + 1) * (l_i + 2) / 2;
    const int n_cartesian_of_l_j = (l_j + 1) * (l_j + 2) / 2;

    for (int i = 0; i < n_cartesian_of_l_i; i++) {
        for (int j = 0; j < n_cartesian_of_l_j; j++) {
            spherical_matrix[d_ao_idx[i_unpad_offset + i_bas * n_cartesian_of_l_i + i] * n_ao_spherical
                           + d_ao_idx[j_unpad_offset + j_bas * n_cartesian_of_l_j + j]]
            = cartesian_matrix[(i_pad_offset + i_bas * n_cartesian_of_l_i + i) * n_ao_cartesian
                             + (j_pad_offset + j_bas * n_cartesian_of_l_j + j)];
        }
    }
}

__global__
static void copy_cartesian_unpad_to_pad(double* cartesian_matrix, const double* spherical_matrix,
                                        const int n_ao_cartesian, const int n_ao_spherical,
                                        const int l_i, const int n_bas_i, const int i_pad_offset, const int i_unpad_offset,
                                        const int l_j, const int n_bas_j, const int j_pad_offset, const int j_unpad_offset,
                                        const int* d_ao_idx)
{
    const int i_bas = blockIdx.x * blockDim.x + threadIdx.x;
    const int j_bas = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_bas >= n_bas_i || j_bas >= n_bas_j)
        return;

    const int n_cartesian_of_l_i = (l_i + 1) * (l_i + 2) / 2;
    const int n_cartesian_of_l_j = (l_j + 1) * (l_j + 2) / 2;

    for (int i = 0; i < n_cartesian_of_l_i; i++) {
        for (int j = 0; j < n_cartesian_of_l_j; j++) {
            cartesian_matrix[(i_pad_offset + i_bas * n_cartesian_of_l_i + i) * n_ao_cartesian
                           + (j_pad_offset + j_bas * n_cartesian_of_l_j + j)]
            = spherical_matrix[d_ao_idx[i_unpad_offset + i_bas * n_cartesian_of_l_i + i] * n_ao_spherical
                             + d_ao_idx[j_unpad_offset + j_bas * n_cartesian_of_l_j + j]];
        }
    }
}

__global__
static void left_cart2cart(double* destination_matrix, const double* source_matrix,
                           const int n_right, const int n_ao_copy, const int i_destination_offset, const int i_source_offset,
                           const int* d_ao_idx)
{
    const int i_right = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_left = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_right >= n_right || i_left >= n_ao_copy)
        return;

    destination_matrix[(i_destination_offset + i_left) * n_right + i_right]
        = source_matrix[d_ao_idx[i_source_offset + i_left] * n_right + i_right];
}

extern "C" {
    // Notice: The cartesian_matrix is used as scratch and the content is destroyed.
    int cart2sph_CT_mat_C_with_padding(const cudaStream_t stream, double* cartesian_matrix, double* spherical_matrix,
                                       const int n_ao_cartesian, const int n_ao_spherical,
                                       const int n_l_ctr_group, const int* l_of_group, const int* n_total_bas_of_group, const int* n_pad_bas_of_group,
                                       const int* d_ao_idx,
                                       const bool if_no_cart2sph)
    {
        if (!if_no_cart2sph) {
            int i_cartesian_offset = 0;
            for (int i_group = 0; i_group < n_l_ctr_group; i_group++) {
                const int l_i = l_of_group[i_group];
                const int n_bas = n_total_bas_of_group[i_group] - n_pad_bas_of_group[i_group];

                const dim3 threads(16, 16);
                const dim3 blocks((n_ao_cartesian + threads.x - 1) / threads.x, (n_bas + threads.y - 1) / threads.y);
                switch (l_i) {
                    case  0: left_cart2sph_inplace< 0> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  1: left_cart2sph_inplace< 1> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  2: left_cart2sph_inplace< 2> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  3: left_cart2sph_inplace< 3> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  4: left_cart2sph_inplace< 4> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  5: left_cart2sph_inplace< 5> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  6: left_cart2sph_inplace< 6> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  7: left_cart2sph_inplace< 7> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  8: left_cart2sph_inplace< 8> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  9: left_cart2sph_inplace< 9> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case 10: left_cart2sph_inplace<10> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    default:
                        printf("l_i = %d not supported for cart2sph_C_mat_CT_with_padding(), max_L = 10\n", l_i);
                        fprintf(stderr, "l_i = %d not supported for cart2sph_C_mat_CT_with_padding(), max_L = 10\n", l_i);
                        return 1;
                }

                i_cartesian_offset += n_total_bas_of_group[i_group] * ((l_i + 1) * (l_i + 2) / 2);
            }

            i_cartesian_offset = 0;
            for (int i_group = 0; i_group < n_l_ctr_group; i_group++) {
                const int l_i = l_of_group[i_group];
                const int n_bas = n_total_bas_of_group[i_group] - n_pad_bas_of_group[i_group];

                const dim3 threads(16, 16);
                const dim3 blocks((n_ao_cartesian + threads.x - 1) / threads.x, (n_bas + threads.y - 1) / threads.y);
                switch (l_i) {
                    case  0: right_cart2sph_inplace< 0> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  1: right_cart2sph_inplace< 1> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  2: right_cart2sph_inplace< 2> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  3: right_cart2sph_inplace< 3> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  4: right_cart2sph_inplace< 4> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  5: right_cart2sph_inplace< 5> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  6: right_cart2sph_inplace< 6> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  7: right_cart2sph_inplace< 7> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  8: right_cart2sph_inplace< 8> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  9: right_cart2sph_inplace< 9> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case 10: right_cart2sph_inplace<10> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    default:
                        printf("l_i = %d not supported for cart2sph_C_mat_CT_with_padding(), max_L = 10\n", l_i);
                        fprintf(stderr, "l_i = %d not supported for cart2sph_C_mat_CT_with_padding(), max_L = 10\n", l_i);
                        return 1;
                }

                i_cartesian_offset += n_total_bas_of_group[i_group] * ((l_i + 1) * (l_i + 2) / 2);
            }

            i_cartesian_offset = 0;
            int i_spherical_offset = 0;
            for (int i_group = 0; i_group < n_l_ctr_group; i_group++) {
                const int l_i = l_of_group[i_group];
                const int n_bas_i = n_total_bas_of_group[i_group] - n_pad_bas_of_group[i_group];
        
                int j_cartesian_offset = 0;
                int j_spherical_offset = 0;
                for (int j_group = 0; j_group < n_l_ctr_group; j_group++) {
                    const int l_j = l_of_group[j_group];
                    const int n_bas_j = n_total_bas_of_group[j_group] - n_pad_bas_of_group[j_group];

                    const dim3 threads(32, 32);
                    const dim3 blocks((n_bas_i + threads.x - 1) / threads.x, (n_bas_j + threads.y - 1) / threads.y);
                    copy_spherical_cart2sph<<<blocks, threads, 0, stream>>>(cartesian_matrix, spherical_matrix, n_ao_cartesian, n_ao_spherical,
                                                                            l_i, n_bas_i, i_cartesian_offset, i_spherical_offset,
                                                                            l_j, n_bas_j, j_cartesian_offset, j_spherical_offset,
                                                                            d_ao_idx);

                    j_cartesian_offset += n_total_bas_of_group[j_group] * ((l_j + 1) * (l_j + 2) / 2);
                    j_spherical_offset += n_bas_j * (l_j * 2 + 1);
                }
                i_cartesian_offset += n_total_bas_of_group[i_group] * ((l_i + 1) * (l_i + 2) / 2);
                i_spherical_offset += n_bas_i * (l_i * 2 + 1);
            }
        } else {
            int i_pad_offset = 0;
            int i_unpad_offset = 0;
            for (int i_group = 0; i_group < n_l_ctr_group; i_group++) {
                const int l_i = l_of_group[i_group];
                const int n_bas_i = n_total_bas_of_group[i_group] - n_pad_bas_of_group[i_group];
        
                int j_pad_offset = 0;
                int j_unpad_offset = 0;
                for (int j_group = 0; j_group < n_l_ctr_group; j_group++) {
                    const int l_j = l_of_group[j_group];
                    const int n_bas_j = n_total_bas_of_group[j_group] - n_pad_bas_of_group[j_group];

                    const dim3 threads(32, 32);
                    const dim3 blocks((n_bas_i + threads.x - 1) / threads.x, (n_bas_j + threads.y - 1) / threads.y);
                    copy_cartesian_pad_to_unpad<<<blocks, threads, 0, stream>>>(cartesian_matrix, spherical_matrix, n_ao_cartesian, n_ao_spherical,
                                                                                l_i, n_bas_i, i_pad_offset, i_unpad_offset,
                                                                                l_j, n_bas_j, j_pad_offset, j_unpad_offset,
                                                                                d_ao_idx);

                    j_pad_offset += n_total_bas_of_group[j_group] * ((l_j + 1) * (l_j + 2) / 2);
                    j_unpad_offset += n_bas_j * ((l_j + 1) * (l_j + 2) / 2);
                }
                i_pad_offset += n_total_bas_of_group[i_group] * ((l_i + 1) * (l_i + 2) / 2);
                i_unpad_offset += n_bas_i * ((l_i + 1) * (l_i + 2) / 2);
            }
        }

        return 0;
    }

    int cart2sph_C_mat_CT_with_padding(const cudaStream_t stream, double* cartesian_matrix, const double* spherical_matrix,
                                       const int n_ao_cartesian, const int n_ao_spherical,
                                       const int n_l_ctr_group, const int* l_of_group, const int* n_total_bas_of_group, const int* n_pad_bas_of_group,
                                       const int* d_ao_idx,
                                       const bool if_no_cart2sph)
    {
        if (!if_no_cart2sph) {
            int i_cartesian_offset = 0;
            int i_spherical_offset = 0;
            for (int i_group = 0; i_group < n_l_ctr_group; i_group++) {
                const int l_i = l_of_group[i_group];
                const int n_bas_i = n_total_bas_of_group[i_group] - n_pad_bas_of_group[i_group];
        
                int j_cartesian_offset = 0;
                int j_spherical_offset = 0;
                for (int j_group = 0; j_group < n_l_ctr_group; j_group++) {
                    const int l_j = l_of_group[j_group];
                    const int n_bas_j = n_total_bas_of_group[j_group] - n_pad_bas_of_group[j_group];

                    const dim3 threads(32, 32);
                    const dim3 blocks((n_bas_i + threads.x - 1) / threads.x, (n_bas_j + threads.y - 1) / threads.y);
                    copy_spherical_sph2cart<<<blocks, threads, 0, stream>>>(cartesian_matrix, spherical_matrix, n_ao_cartesian, n_ao_spherical,
                                                                            l_i, n_bas_i, i_cartesian_offset, i_spherical_offset,
                                                                            l_j, n_bas_j, j_cartesian_offset, j_spherical_offset,
                                                                            d_ao_idx);

                    j_cartesian_offset += n_total_bas_of_group[j_group] * ((l_j + 1) * (l_j + 2) / 2);
                    j_spherical_offset += n_bas_j * (l_j * 2 + 1);
                }
                i_cartesian_offset += n_total_bas_of_group[i_group] * ((l_i + 1) * (l_i + 2) / 2);
                i_spherical_offset += n_bas_i * (l_i * 2 + 1);
            }

            i_cartesian_offset = 0;
            for (int i_group = 0; i_group < n_l_ctr_group; i_group++) {
                const int l_i = l_of_group[i_group];
                const int n_bas = n_total_bas_of_group[i_group] - n_pad_bas_of_group[i_group];

                const dim3 threads(16, 16);
                const dim3 blocks((n_ao_cartesian + threads.x - 1) / threads.x, (n_bas + threads.y - 1) / threads.y);
                switch (l_i) {
                    case  0: left_sph2cart_inplace< 0> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  1: left_sph2cart_inplace< 1> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  2: left_sph2cart_inplace< 2> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  3: left_sph2cart_inplace< 3> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  4: left_sph2cart_inplace< 4> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  5: left_sph2cart_inplace< 5> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  6: left_sph2cart_inplace< 6> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  7: left_sph2cart_inplace< 7> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  8: left_sph2cart_inplace< 8> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  9: left_sph2cart_inplace< 9> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case 10: left_sph2cart_inplace<10> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    default:
                        printf("l_i = %d not supported for cart2sph_CT_mat_C_with_padding(), max_L = 10\n", l_i);
                        fprintf(stderr, "l_i = %d not supported for cart2sph_CT_mat_C_with_padding(), max_L = 10\n", l_i);
                        return 1;
                }

                i_cartesian_offset += n_total_bas_of_group[i_group] * ((l_i + 1) * (l_i + 2) / 2);
            }

            i_cartesian_offset = 0;
            for (int i_group = 0; i_group < n_l_ctr_group; i_group++) {
                const int l_i = l_of_group[i_group];
                const int n_bas = n_total_bas_of_group[i_group] - n_pad_bas_of_group[i_group];

                const dim3 threads(16, 16);
                const dim3 blocks((n_ao_cartesian + threads.x - 1) / threads.x, (n_bas + threads.y - 1) / threads.y);
                switch (l_i) {
                    case  0: right_sph2cart_inplace< 0> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  1: right_sph2cart_inplace< 1> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  2: right_sph2cart_inplace< 2> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  3: right_sph2cart_inplace< 3> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  4: right_sph2cart_inplace< 4> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  5: right_sph2cart_inplace< 5> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  6: right_sph2cart_inplace< 6> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  7: right_sph2cart_inplace< 7> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  8: right_sph2cart_inplace< 8> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case  9: right_sph2cart_inplace< 9> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    case 10: right_sph2cart_inplace<10> <<<blocks, threads, 0, stream>>>(cartesian_matrix, n_ao_cartesian, n_bas, i_cartesian_offset); break;
                    default:
                        printf("l_i = %d not supported for cart2sph_CT_mat_C_with_padding(), max_L = 10\n", l_i);
                        fprintf(stderr, "l_i = %d not supported for cart2sph_CT_mat_C_with_padding(), max_L = 10\n", l_i);
                        return 1;
                }

                i_cartesian_offset += n_total_bas_of_group[i_group] * ((l_i + 1) * (l_i + 2) / 2);
            }
        } else {
            int i_pad_offset = 0;
            int i_unpad_offset = 0;
            for (int i_group = 0; i_group < n_l_ctr_group; i_group++) {
                const int l_i = l_of_group[i_group];
                const int n_bas_i = n_total_bas_of_group[i_group] - n_pad_bas_of_group[i_group];
        
                int j_pad_offset = 0;
                int j_unpad_offset = 0;
                for (int j_group = 0; j_group < n_l_ctr_group; j_group++) {
                    const int l_j = l_of_group[j_group];
                    const int n_bas_j = n_total_bas_of_group[j_group] - n_pad_bas_of_group[j_group];

                    const dim3 threads(32, 32);
                    const dim3 blocks((n_bas_i + threads.x - 1) / threads.x, (n_bas_j + threads.y - 1) / threads.y);
                    copy_cartesian_unpad_to_pad<<<blocks, threads, 0, stream>>>(cartesian_matrix, spherical_matrix, n_ao_cartesian, n_ao_spherical,
                                                                                l_i, n_bas_i, i_pad_offset, i_unpad_offset,
                                                                                l_j, n_bas_j, j_pad_offset, j_unpad_offset,
                                                                                d_ao_idx);

                    j_pad_offset += n_total_bas_of_group[j_group] * ((l_j + 1) * (l_j + 2) / 2);
                    j_unpad_offset += n_bas_j * ((l_j + 1) * (l_j + 2) / 2);
                }
                i_pad_offset += n_total_bas_of_group[i_group] * ((l_i + 1) * (l_i + 2) / 2);
                i_unpad_offset += n_bas_i * ((l_i + 1) * (l_i + 2) / 2);
            }
        }

        return 0;
    }

    int cart2sph_C_mat_with_padding(const cudaStream_t stream, double* cartesian_matrix, const double* spherical_matrix,
                                    const int n_right,
                                    const int n_l_ctr_group, const int* l_of_group, const int* n_total_bas_of_group, const int* n_pad_bas_of_group,
                                    const int* d_ao_idx,
                                    const bool if_no_cart2sph)
    {
        if (!if_no_cart2sph) {
            int i_cartesian_offset = 0;
            int i_spherical_offset = 0;
            for (int i_group = 0; i_group < n_l_ctr_group; i_group++) {
                const int l_i = l_of_group[i_group];
                const int n_bas = n_total_bas_of_group[i_group] - n_pad_bas_of_group[i_group];

                const dim3 threads(16, 16);
                const dim3 blocks((n_right + threads.x - 1) / threads.x, (n_bas + threads.y - 1) / threads.y);
                switch (l_i) {
                    case  0: left_sph2cart< 0> <<<blocks, threads, 0, stream>>>(cartesian_matrix, spherical_matrix, n_right, n_bas, i_cartesian_offset, i_spherical_offset, d_ao_idx); break;
                    case  1: left_sph2cart< 1> <<<blocks, threads, 0, stream>>>(cartesian_matrix, spherical_matrix, n_right, n_bas, i_cartesian_offset, i_spherical_offset, d_ao_idx); break;
                    case  2: left_sph2cart< 2> <<<blocks, threads, 0, stream>>>(cartesian_matrix, spherical_matrix, n_right, n_bas, i_cartesian_offset, i_spherical_offset, d_ao_idx); break;
                    case  3: left_sph2cart< 3> <<<blocks, threads, 0, stream>>>(cartesian_matrix, spherical_matrix, n_right, n_bas, i_cartesian_offset, i_spherical_offset, d_ao_idx); break;
                    case  4: left_sph2cart< 4> <<<blocks, threads, 0, stream>>>(cartesian_matrix, spherical_matrix, n_right, n_bas, i_cartesian_offset, i_spherical_offset, d_ao_idx); break;
                    case  5: left_sph2cart< 5> <<<blocks, threads, 0, stream>>>(cartesian_matrix, spherical_matrix, n_right, n_bas, i_cartesian_offset, i_spherical_offset, d_ao_idx); break;
                    case  6: left_sph2cart< 6> <<<blocks, threads, 0, stream>>>(cartesian_matrix, spherical_matrix, n_right, n_bas, i_cartesian_offset, i_spherical_offset, d_ao_idx); break;
                    case  7: left_sph2cart< 7> <<<blocks, threads, 0, stream>>>(cartesian_matrix, spherical_matrix, n_right, n_bas, i_cartesian_offset, i_spherical_offset, d_ao_idx); break;
                    case  8: left_sph2cart< 8> <<<blocks, threads, 0, stream>>>(cartesian_matrix, spherical_matrix, n_right, n_bas, i_cartesian_offset, i_spherical_offset, d_ao_idx); break;
                    case  9: left_sph2cart< 9> <<<blocks, threads, 0, stream>>>(cartesian_matrix, spherical_matrix, n_right, n_bas, i_cartesian_offset, i_spherical_offset, d_ao_idx); break;
                    case 10: left_sph2cart<10> <<<blocks, threads, 0, stream>>>(cartesian_matrix, spherical_matrix, n_right, n_bas, i_cartesian_offset, i_spherical_offset, d_ao_idx); break;
                    default:
                        printf("l_i = %d not supported for cart2sph_C_mat_with_padding(), max_L = 10\n", l_i);
                        fprintf(stderr, "l_i = %d not supported for cart2sph_C_mat_with_padding(), max_L = 10\n", l_i);
                        return 1;
                }

                i_cartesian_offset += n_total_bas_of_group[i_group] * ((l_i + 1) * (l_i + 2) / 2);
                i_spherical_offset += n_bas * (l_i * 2 + 1);
            }
        } else {
            int i_pad_offset = 0;
            int i_unpad_offset = 0;
            for (int i_group = 0; i_group < n_l_ctr_group; i_group++) {
                const int l_i = l_of_group[i_group];
                const int n_bas = n_total_bas_of_group[i_group] - n_pad_bas_of_group[i_group];
                const int n_cartesian_of_l = (l_i + 1) * (l_i + 2) / 2;

                const dim3 threads(16, 16);
                const dim3 blocks((n_right + threads.x - 1) / threads.x, (n_bas * n_cartesian_of_l + threads.y - 1) / threads.y);
                left_cart2cart<<<threads, blocks>>>(cartesian_matrix, spherical_matrix,
                                                    n_right, n_bas * n_cartesian_of_l, i_pad_offset, i_unpad_offset, d_ao_idx);

                i_pad_offset += n_total_bas_of_group[i_group] * n_cartesian_of_l;
                i_unpad_offset += n_bas * n_cartesian_of_l;
            }
        }

        return 0;
    }
}
