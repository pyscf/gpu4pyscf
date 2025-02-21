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
#define AO_LMAX         4

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
        //10,9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
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
        //0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
        //0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
        //0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
        //0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
        //0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
};

__constant__
static int _offset_cart[] = {0, 1, 4, 10, 20, 35, 56, 84, 120, 165};

__constant__
static double _binom[] = {
        1,
        1, 1,
        1, 2, 1,
        1, 3, 3, 1,
        1, 4, 6, 4, 1,
        1, 5, 10, 10, 5, 1,};

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
