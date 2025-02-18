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

template <int L> __device__
static void cart2sph(double *gsph, double *gcart){
    if (L == 0) {
        gsph[0] = 0.282094791773878143 * gcart[0];
    }

    if (L == 1) {
        gsph[0] = 0.488602511902919921 * gcart[0];
        gsph[1] = 0.488602511902919921 * gcart[1];
        gsph[2] = 0.488602511902919921 * gcart[2];
    }

    if (L == 2) {
        gsph[0] = 1.092548430592079070 * gcart[1];
        gsph[1] = 1.092548430592079070 * gcart[4];
        gsph[2] = 0.630783130505040012 * gcart[5] - 0.315391565252520002 * (gcart[0] + gcart[3]);
        gsph[3] = 1.092548430592079070 * gcart[2];
        gsph[4] = 0.546274215296039535 * (gcart[0] - gcart[3]);
    }

    if (L == 3) {
        gsph[0] = 1.7701307697799304*gcart[1] + -0.5900435899266435*gcart[6];
        gsph[1] = 2.8906114426405543*gcart[4];
        gsph[2] = -0.4570457994644657*gcart[1] + -0.4570457994644657*gcart[6] + 1.8281831978578629*gcart[8];
        gsph[3] = -1.1195289977703462*gcart[2] + -1.1195289977703462*gcart[7] + 0.7463526651802308*gcart[9];
        gsph[4] = -0.4570457994644657*gcart[0] + -0.4570457994644657*gcart[3] + 1.8281831978578629*gcart[5];
        gsph[5] = 1.4453057213202771*gcart[2] + -1.4453057213202771*gcart[7];
        gsph[6] = 0.5900435899266435*gcart[0] + -1.7701307697799304*gcart[3];
    }

    if (L == 4) {
        gsph[0] = 2.5033429417967046*gcart[1] + -2.5033429417967046*gcart[6];
        gsph[1] = 5.310392309339791*gcart[4] + -1.7701307697799304*gcart[11];
        gsph[2] = -0.94617469575756*gcart[1] + -0.94617469575756*gcart[6] + 5.6770481745453605*gcart[8];
        gsph[3] = -2.0071396306718676*gcart[4] + -2.0071396306718676*gcart[11] + 2.676186174229157*gcart[13];
        gsph[4] = 0.31735664074561293*gcart[0] + 0.6347132814912259*gcart[3] + -2.5388531259649034*gcart[5] + 0.31735664074561293*gcart[10] + -2.5388531259649034*gcart[12] + 0.8462843753216345*gcart[14];
        gsph[5] = -2.0071396306718676*gcart[2] + -2.0071396306718676*gcart[7] + 2.676186174229157*gcart[9];
        gsph[6] = -0.47308734787878*gcart[0] + 2.8385240872726802*gcart[5] + 0.47308734787878*gcart[10] + -2.8385240872726802*gcart[12];
        gsph[7] = 1.7701307697799304*gcart[2] + -5.310392309339791*gcart[7];
        gsph[8] = 0.6258357354491761*gcart[0] + -3.755014412695057*gcart[3] + 0.6258357354491761*gcart[10];
    }

    if (L == 5) {
        gsph[0] = 3.2819102842008507*gcart[1] + -6.563820568401701*gcart[6] + 0.6563820568401701*gcart[15];
        gsph[1] = 8.302649259524165*gcart[4] + -8.302649259524165*gcart[11];
        gsph[2] = -1.467714898305751*gcart[1] + -0.9784765988705008*gcart[6] + 11.741719186446009*gcart[8] + 0.4892382994352504*gcart[15] + -3.913906395482003*gcart[17];
        gsph[3] = -4.793536784973324*gcart[4] + -4.793536784973324*gcart[11] + 9.587073569946648*gcart[13];
        gsph[4] = 0.45294665119569694*gcart[1] + 0.9058933023913939*gcart[6] + -5.435359814348363*gcart[8] + 0.45294665119569694*gcart[15] + -5.435359814348363*gcart[17] + 3.6235732095655755*gcart[19];
        gsph[5] = 1.754254836801354*gcart[2] + 3.508509673602708*gcart[7] + -4.678012898136944*gcart[9] + 1.754254836801354*gcart[16] + -4.678012898136944*gcart[18] + 0.9356025796273888*gcart[20];
        gsph[6] = 0.45294665119569694*gcart[0] + 0.9058933023913939*gcart[3] + -5.435359814348363*gcart[5] + 0.45294665119569694*gcart[10] + -5.435359814348363*gcart[12] + 3.6235732095655755*gcart[14];
        gsph[7] = -2.396768392486662*gcart[2] + 4.793536784973324*gcart[9] + 2.396768392486662*gcart[16] + -4.793536784973324*gcart[18];
        gsph[8] = -0.4892382994352504*gcart[0] + 0.9784765988705008*gcart[3] + 3.913906395482003*gcart[5] + 1.467714898305751*gcart[10] + -11.741719186446009*gcart[12];
        gsph[9] = 2.075662314881041*gcart[2] + -12.453973889286248*gcart[7] + 2.075662314881041*gcart[16];
        gsph[10] = 0.6563820568401701*gcart[0] + -6.563820568401701*gcart[3] + 3.2819102842008507*gcart[10];
    }

    if (L == 6) {
        gsph[0] = 4.099104631151486*gcart[1] + -13.663682103838289*gcart[6] + 4.099104631151486*gcart[15];
        gsph[1] = 11.833095811158763*gcart[4] + -23.666191622317527*gcart[11] + 2.3666191622317525*gcart[22];
        gsph[2] = -2.0182596029148963*gcart[1] + 20.182596029148968*gcart[8] + 2.0182596029148963*gcart[15] + -20.182596029148968*gcart[17];
        gsph[3] = -8.29084733563431*gcart[4] + -5.527231557089541*gcart[11] + 22.108926228358165*gcart[13] + 2.7636157785447706*gcart[22] + -7.369642076119389*gcart[24];
        gsph[4] = 0.9212052595149236*gcart[1] + 1.8424105190298472*gcart[6] + -14.739284152238778*gcart[8] + 0.9212052595149236*gcart[15] + -14.739284152238778*gcart[17] + 14.739284152238778*gcart[19];
        gsph[5] = 2.913106812593657*gcart[4] + 5.826213625187314*gcart[11] + -11.652427250374627*gcart[13] + 2.913106812593657*gcart[22] + -11.652427250374627*gcart[24] + 4.6609709001498505*gcart[26];
        gsph[6] = -0.3178460113381421*gcart[0] + -0.9535380340144264*gcart[3] + 5.721228204086558*gcart[5] + -0.9535380340144264*gcart[10] + 11.442456408173117*gcart[12] + -7.628304272115411*gcart[14] + -0.3178460113381421*gcart[21] + 5.721228204086558*gcart[23] + -7.628304272115411*gcart[25] + 1.0171072362820548*gcart[27];
        gsph[7] = 2.913106812593657*gcart[2] + 5.826213625187314*gcart[7] + -11.652427250374627*gcart[9] + 2.913106812593657*gcart[16] + -11.652427250374627*gcart[18] + 4.6609709001498505*gcart[20];
        gsph[8] = 0.4606026297574618*gcart[0] + 0.4606026297574618*gcart[3] + -7.369642076119389*gcart[5] + -0.4606026297574618*gcart[10] + 7.369642076119389*gcart[14] + -0.4606026297574618*gcart[21] + 7.369642076119389*gcart[23] + -7.369642076119389*gcart[25];
        gsph[9] = -2.7636157785447706*gcart[2] + 5.527231557089541*gcart[7] + 7.369642076119389*gcart[9] + 8.29084733563431*gcart[16] + -22.108926228358165*gcart[18];
        gsph[10] = -0.5045649007287241*gcart[0] + 2.52282450364362*gcart[3] + 5.045649007287242*gcart[5] + 2.52282450364362*gcart[10] + -30.273894043723452*gcart[12] + -0.5045649007287241*gcart[21] + 5.045649007287242*gcart[23];
        gsph[11] = 2.3666191622317525*gcart[2] + -23.666191622317527*gcart[7] + 11.833095811158763*gcart[16];
        gsph[12] = 0.6831841051919144*gcart[0] + -10.247761577878716*gcart[3] + 10.247761577878716*gcart[10] + -0.6831841051919144*gcart[21];
    }

    if(L == 7) {
        gsph[0] = 4.950139127672174*gcart[1] + -24.75069563836087*gcart[6] + 14.850417383016522*gcart[15] + -0.7071627325245963*gcart[28];
        gsph[1] = 15.8757639708114*gcart[4] + -52.919213236038004*gcart[11] + 15.8757639708114*gcart[22];
        gsph[2] = -2.594577893601302*gcart[1] + 2.594577893601302*gcart[6] + 31.134934723215622*gcart[8] + 4.670240208482344*gcart[15] + -62.269869446431244*gcart[17] + -0.5189155787202604*gcart[28] + 6.226986944643125*gcart[30];
        gsph[3] = -12.45397388928625*gcart[4] + 41.51324629762083*gcart[13] + 12.45397388928625*gcart[22] + -41.51324629762083*gcart[24];
        gsph[4] = 1.4081304047606462*gcart[1] + 2.3468840079344107*gcart[6] + -28.162608095212924*gcart[8] + 0.4693768015868821*gcart[15] + -18.77507206347528*gcart[17] + 37.55014412695057*gcart[19] + -0.4693768015868821*gcart[28] + 9.38753603173764*gcart[30] + -12.516714708983523*gcart[32];
        gsph[5] = 6.637990386674741*gcart[4] + 13.275980773349483*gcart[11] + -35.402615395598616*gcart[13] + 6.637990386674741*gcart[22] + -35.402615395598616*gcart[24] + 21.241569237359172*gcart[26];
        gsph[6] = -0.4516580379125866*gcart[1] + -1.35497411373776*gcart[6] + 10.839792909902078*gcart[8] + -1.35497411373776*gcart[15] + 21.679585819804156*gcart[17] + -21.679585819804156*gcart[19] + -0.4516580379125866*gcart[28] + 10.839792909902078*gcart[30] + -21.679585819804156*gcart[32] + 5.781222885281109*gcart[34];
        gsph[7] = -2.389949691920173*gcart[2] + -7.169849075760519*gcart[7] + 14.339698151521036*gcart[9] + -7.169849075760519*gcart[16] + 28.679396303042072*gcart[18] + -11.47175852121683*gcart[20] + -2.389949691920173*gcart[29] + 14.339698151521036*gcart[31] + -11.47175852121683*gcart[33] + 1.092548430592079*gcart[35];
        gsph[8] = -0.4516580379125866*gcart[0] + -1.35497411373776*gcart[3] + 10.839792909902078*gcart[5] + -1.35497411373776*gcart[10] + 21.679585819804156*gcart[12] + -21.679585819804156*gcart[14] + -0.4516580379125866*gcart[21] + 10.839792909902078*gcart[23] + -21.679585819804156*gcart[25] + 5.781222885281109*gcart[27];
        gsph[9] = 3.3189951933373707*gcart[2] + 3.3189951933373707*gcart[7] + -17.701307697799308*gcart[9] + -3.3189951933373707*gcart[16] + 10.620784618679586*gcart[20] + -3.3189951933373707*gcart[29] + 17.701307697799308*gcart[31] + -10.620784618679586*gcart[33];
        gsph[10] = 0.4693768015868821*gcart[0] + -0.4693768015868821*gcart[3] + -9.38753603173764*gcart[5] + -2.3468840079344107*gcart[10] + 18.77507206347528*gcart[12] + 12.516714708983523*gcart[14] + -1.4081304047606462*gcart[21] + 28.162608095212924*gcart[23] + -37.55014412695057*gcart[25];
        gsph[11] = -3.1134934723215624*gcart[2] + 15.567467361607811*gcart[7] + 10.378311574405208*gcart[9] + 15.567467361607811*gcart[16] + -62.269869446431244*gcart[18] + -3.1134934723215624*gcart[29] + 10.378311574405208*gcart[31];
        gsph[12] = -0.5189155787202604*gcart[0] + 4.670240208482344*gcart[3] + 6.226986944643125*gcart[5] + 2.594577893601302*gcart[10] + -62.269869446431244*gcart[12] + -2.594577893601302*gcart[21] + 31.134934723215622*gcart[23];
        gsph[13] = 2.6459606618019*gcart[2] + -39.6894099270285*gcart[7] + 39.6894099270285*gcart[16] + -2.6459606618019*gcart[29];
        gsph[14] = 0.7071627325245963*gcart[0] + -14.850417383016522*gcart[3] + 24.75069563836087*gcart[10] + -4.950139127672174*gcart[21];
    }

    if(L == 8){
        gsph[0] = 5.83141328139864*gcart[1] + -40.81989296979048*gcart[6] + 40.81989296979048*gcart[15] + -5.83141328139864*gcart[28];
        gsph[1] = 20.40994648489524*gcart[4] + -102.0497324244762*gcart[11] + 61.22983945468572*gcart[22] + -2.91570664069932*gcart[37];
        gsph[2] = -3.193996596357255*gcart[1] + 7.452658724833595*gcart[6] + 44.71595234900157*gcart[8] + 7.452658724833595*gcart[15] + -149.0531744966719*gcart[17] + -3.193996596357255*gcart[28] + 44.71595234900157*gcart[30];
        gsph[3] = -17.24955311049054*gcart[4] + 17.24955311049054*gcart[11] + 68.99821244196217*gcart[13] + 31.04919559888297*gcart[22] + -137.9964248839243*gcart[24] + -3.449910622098108*gcart[37] + 13.79964248839243*gcart[39];
        gsph[4] = 1.913666099037323*gcart[1] + 1.913666099037323*gcart[6] + -45.92798637689575*gcart[8] + -1.913666099037323*gcart[15] + 76.54664396149292*gcart[19] + -1.913666099037323*gcart[28] + 45.92798637689575*gcart[30] + -76.54664396149292*gcart[32];
        gsph[5] = 11.1173953976599*gcart[4] + 18.52899232943316*gcart[11] + -74.11596931773265*gcart[13] + 3.705798465886632*gcart[22] + -49.41064621182176*gcart[24] + 59.29277545418611*gcart[26] + -3.705798465886632*gcart[37] + 24.70532310591088*gcart[39] + -19.7642584847287*gcart[41];
        gsph[6] = -0.912304516869819*gcart[1] + -2.736913550609457*gcart[6] + 27.36913550609457*gcart[8] + -2.736913550609457*gcart[15] + 54.73827101218914*gcart[17] + -72.98436134958553*gcart[19] + -0.912304516869819*gcart[28] + 27.36913550609457*gcart[30] + -72.98436134958553*gcart[32] + 29.19374453983421*gcart[34];
        gsph[7] = -3.8164436064573*gcart[4] + -11.4493308193719*gcart[11] + 30.5315488516584*gcart[13] + -11.4493308193719*gcart[22] + 61.06309770331679*gcart[24] + -36.63785862199007*gcart[26] + -3.8164436064573*gcart[37] + 30.5315488516584*gcart[39] + -36.63785862199007*gcart[41] + 6.978639737521918*gcart[43];
        gsph[8] = 0.3180369672047749*gcart[0] + 1.272147868819099*gcart[3] + -10.1771829505528*gcart[5] + 1.908221803228649*gcart[10] + -30.53154885165839*gcart[12] + 30.53154885165839*gcart[14] + 1.272147868819099*gcart[21] + -30.53154885165839*gcart[23] + 61.06309770331677*gcart[25] + -16.28349272088447*gcart[27] + 0.3180369672047749*gcart[36] + -10.1771829505528*gcart[38] + 30.53154885165839*gcart[40] + -16.28349272088447*gcart[42] + 1.16310662292032*gcart[44];
        gsph[9] = -3.8164436064573*gcart[2] + -11.4493308193719*gcart[7] + 30.5315488516584*gcart[9] + -11.4493308193719*gcart[16] + 61.06309770331679*gcart[18] + -36.63785862199007*gcart[20] + -3.8164436064573*gcart[29] + 30.5315488516584*gcart[31] + -36.63785862199007*gcart[33] + 6.978639737521918*gcart[35];
        gsph[10] = -0.4561522584349095*gcart[0] + -0.912304516869819*gcart[3] + 13.68456775304729*gcart[5] + 13.68456775304729*gcart[12] + -36.49218067479276*gcart[14] + 0.912304516869819*gcart[21] + -13.68456775304729*gcart[23] + 14.5968722699171*gcart[27] + 0.4561522584349095*gcart[36] + -13.68456775304729*gcart[38] + 36.49218067479276*gcart[40] + -14.5968722699171*gcart[42];
        gsph[11] = 3.705798465886632*gcart[2] + -3.705798465886632*gcart[7] + -24.70532310591088*gcart[9] + -18.52899232943316*gcart[16] + 49.41064621182176*gcart[18] + 19.7642584847287*gcart[20] + -11.1173953976599*gcart[29] + 74.11596931773265*gcart[31] + -59.29277545418611*gcart[33];
        gsph[12] = 0.4784165247593308*gcart[0] + -1.913666099037323*gcart[3] + -11.48199659422394*gcart[5] + -4.784165247593307*gcart[10] + 57.40998297111968*gcart[12] + 19.13666099037323*gcart[14] + -1.913666099037323*gcart[21] + 57.40998297111968*gcart[23] + -114.8199659422394*gcart[25] + 0.4784165247593308*gcart[36] + -11.48199659422394*gcart[38] + 19.13666099037323*gcart[40];
        gsph[13] = -3.449910622098108*gcart[2] + 31.04919559888297*gcart[7] + 13.79964248839243*gcart[9] + 17.24955311049054*gcart[16] + -137.9964248839243*gcart[18] + -17.24955311049054*gcart[29] + 68.99821244196217*gcart[31];
        gsph[14] = -0.5323327660595425*gcart[0] + 7.452658724833595*gcart[3] + 7.452658724833595*gcart[5] + -111.7898808725039*gcart[12] + -7.452658724833595*gcart[21] + 111.7898808725039*gcart[23] + 0.5323327660595425*gcart[36] + -7.452658724833595*gcart[38];
        gsph[15] = 2.91570664069932*gcart[2] + -61.22983945468572*gcart[7] + 102.0497324244762*gcart[16] + -20.40994648489524*gcart[29];
        gsph[16] = 0.72892666017483*gcart[0] + -20.40994648489524*gcart[3] + 51.0248662122381*gcart[10] + -20.40994648489524*gcart[21] + 0.72892666017483*gcart[36];
    }
}
