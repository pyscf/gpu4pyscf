#define EXPCUTOFF       39   // 1e-17
#define CUTOFF          460  // ~ 1e200
#define CART_CUM        (165)

// slots of bas
#define ATOM_OF         0
#define ANG_OF          1
#define NPRIM_OF        2
#define NCTR_OF         3
#define KAPPA_OF        4
#define PTR_EXP         5
#define PTR_COEFF       6
#define PTR_BAS_COORD   7
#define BAS_SLOTS       8

#define RADI_POWER      3
#define SO_TYPE_OF      4

// atm
#define PTR_COORD       1
#define ATM_SLOTS       6

#define ECP_LMAX        4
#define AO_LMAX         4    // Up to G
#define AO_LMAX_IP      6    // Up to G, and its second derivative
#define NF_MAX          15
#define AO_LIJMAX       10
#define NF_MAX_LIJ      66    // Up to l=10

// Thread number has to be the same as qudrature points
#define THREADS        128

__constant__
static int _cart_pow_y[] = {
        0,
        1, 0,
        2, 1, 0,
        3, 2, 1, 0,
        4, 3, 2, 1, 0,
        5, 4, 3, 2, 1, 0,
        6, 5, 4, 3, 2, 1, 0,
        7, 6, 5, 4, 3, 2, 1, 0,
        8, 7, 6, 5, 4, 3, 2, 1, 0,
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        10,9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        //11,10,9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        //12,11,10,9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        //13,12,11,10,9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        //14,13,12,11,10,9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
};

__constant__
static int _cart_pow_z[] = {
        0,
        0, 1,
        0, 1, 2,
        0, 1, 2, 3,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5, 6,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        //0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        //0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        //0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
        //0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
};

//__constant__
//static int _offset_cart[] = {0, 1, 4, 10, 20, 35, 56, 84, 120, 165};

__constant__
static double _binom[] = {
        1,
        1, 1,
        1, 2, 1,
        1, 3, 3, 1,
        1, 4, 6, 4, 1,
        1, 5, 10, 10, 5, 1,
        1, 6, 15, 20, 15, 6, 1,};

__constant__
static int _y_addr[] = {
  1,   // l = 0
  3,   4,    // l = 1
  6,   7,   8,   // l = 2
  10,  11,  12,  13,   // l = 3
  15,  16,  17,  18,  19, // l = 4
  21,  22,  23,  24,  25,  26,   // l = 5
  28,  29,  30,  31,  32,  33,  34,   // l = 6
  36,  37,  38,  39,  40,  41,  42,  43,   // l = 7
  45,  46,  47,  48,  49,  50,  51,  52,  53,  // l = 8
  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,   // l = 9
  //66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,
  //78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
  //91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
  //105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
  //120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
};

__constant__
static int _z_addr[] = {
  2,   // l = 0
  4,   5,    // l = 1
  7,   8,   9,   // l = 2
  11,  12,  13,  14,   // l = 3
  16,  17,  18,  19,  20, // l = 4
  22,  23,  24,  25,  26,  27,   // l = 5
  29,  30,  31,  32,  33,  34,  35,   // l = 6
  37,  38,  39,  40,  41,  42,  43,  44,   // l = 7
  46,  47,  48,  49,  50,  51,  52,  53,  54,   // l = 8
  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,   // l = 9
  //67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
  //79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
  //92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104,
  //106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
  //121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
};

__constant__
static double _common_fac[] = {
    0.282094791773878143,
    0.488602511902919921,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
};

__constant__
static double _ecp_fac[] = {
    0.5773502691896258,
    0.488602511902919921,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
};
