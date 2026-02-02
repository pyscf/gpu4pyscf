#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <memory>
#include <optional>
#include <algorithm>
#include <cctype>
#include <optional>
#include <string_view>

#include "gint/sycl_device.hpp"
#include <exchcxx/exchcxx.hpp>
#include <atomic>
#include <mutex>
#include "exchcxx.h"                     // ABI structs

namespace detail {

  static std::string to_upper(std::string s){
    for(auto &c : s) c = char(std::toupper(unsigned(c)));
    return s;
  }

  // Fast path: look up by canonical ExchCXX functional name (already in functional_map)
  inline std::optional<ExchCXX::Functional>
  functional_from_string(std::string_view s) {
    const auto key = to_upper(std::string{s});
    try {
      return ExchCXX::functional_map.value(key);  // name -> enum
    } catch (const std::out_of_range&) {
      return std::nullopt;                         // not present
    }
  }

  // Minimal alias map: LibXC “family labels” → ExchCXX canonical name
  static const std::unordered_map<std::string,std::string> kLibXCAliases = {
    // Hybrids / composites
    {"HYB_GGA_XC_B3LYP",   "B3LYP"}, // this is an issue since B3LYP uses VWN5 varient, but looks like pyscf 2.3.0> uses B3LYP VWN_RPA version
    {"HYB_GGA_XC_PBEH",    "PBE0"}, // ok
    {"HYB_GGA_XC_HSE03",   "HSE03"}, // ok
    {"HYB_GGA_XC_HSE06",   "HSE06"}, // ok
    {"HYB_GGA_XC_CAM_B3LYP","CAMB3LYP"}, // ok
    {"HYB_MGGA_X_SCAN0",   "SCAN0"},
    {"HYB_GGA_XC_B3PW91",  "B3PW91"}, // ok
    {"HYB_GGA_XC_BHANDH",  "BHANDH"},
    {"HYB_GGA_XC_O3LYP",   "O3LYP"}, // ok

    // Pure composites
    {"GGA_XC_PBE",         "PBE"},
    {"GGA_XC_REVPBE",      "REVPBE"},
    {"GGA_XC_BLYP",        "BLYP"}, // needs checking
    {"GGA_XC_BP86",        "BP86"},
    {"GGA_XC_PW91",        "PW91"}, // may be incorrect
    {"GGA_XC_RPBE",        "RPBE"},
    {"GGA_XC_X3LYP",       "X3LYP"},
    {"GGA_XC_XLYP",        "XLYP"},
    {"GGA_XC_OPBE",        "OPBE"}, // incorrect
    {"GGA_XC_OLYP",        "OLYP"}, // incorrect

    // mGGA composites
    {"MGGA_XC_SCAN",       "SCAN"},
    {"MGGA_XC_R2SCAN",     "R2SCAN"},
    {"MGGA_XC_TPSS",       "TPSS"},
    {"MGGA_XC_REVTPSS",    "REVTPSS"},
    {"MGGA_XC_M06_L",      "M06L"},

    // LDA packs
    {"LDA_XC_VWN",         "SPW92"}, // LibXC’s “VWN” combo equals Slater + VWN
    {"LDA_XC_SVWN",        "SVWN5"}, // Many LibXC builds alias SVWN→SVWN5
    {"LDA_XC_SVWN3",       "SVWN3"},
    {"LDA_XC_SVWN5",       "SVWN5"},
    {"LDA_XC_LDA",         "LDA"},
  };

  // Robust LibXC → ExchCXX functional resolution.
  // 1) Direct hit in alias table
  // 2) Heuristics for common pairs (e.g. GGA_X_PBE + GGA_C_PBE ⇒ PBE) can be handled
  //    by your caller if it sees both half-labels; for single labels use the alias table.
  inline std::optional<ExchCXX::Functional>
  libxc_name_to_functional(std::string_view libxc_name) {
    std::cout << "string name from libxc_name_to_functional() : " << libxc_name << std::endl;
    auto key = to_upper(std::string{libxc_name});

    // 1) Exact alias → canonical functional name
    if (auto it = kLibXCAliases.find(key); it != kLibXCAliases.end()) {
      if (auto f = functional_from_string(it->second)) return f;
    }

    // 2) Loose pattern matches for families (cheap and safe)
    // CAM-B3LYP spellings vary a bit across LibXC versions
    if (key.find("CAM") != std::string::npos && key.find("B3LYP") != std::string::npos) {
      if (auto f = functional_from_string("CAMB3LYP")) return f;
    }
    // LRC-ωPBE family
    if (key.find("LC_WPBE") != std::string::npos || key.find("LRC_WPBE") != std::string::npos) {
      if (key.find('H') != std::string::npos) {
        if (auto f = functional_from_string("LRCWPBEH")) return f;
      } else {
        if (auto f = functional_from_string("LRCWPBE"))  return f;
      }
    }

    // 3) Already a canonical ExchCXX name? (users sometimes pass that)
    if (auto f = functional_from_string(key)) return f;

    return std::nullopt;
  }

  std::unordered_map< ExchCXX::Kernel, std::string > libxc_kernel_map {
    // LDA Functionals
    { ExchCXX::Kernel::SlaterExchange, "LDA_X"             },
    { ExchCXX::Kernel::VWN3,           "LDA_C_VWN_3"       },
    { ExchCXX::Kernel::VWN5,           "LDA_C_VWN_RPA"     },
    { ExchCXX::Kernel::VWN,            "LDA_C_VWN"         },
    { ExchCXX::Kernel::PZ81,           "LDA_C_PZ"          },
    { ExchCXX::Kernel::PZ81_MOD,       "LDA_C_PZ_MOD"      },
    { ExchCXX::Kernel::PW91_LDA,       "LDA_C_PW"          },
    { ExchCXX::Kernel::PW91_LDA_MOD,   "LDA_C_PW_MOD"      },
    { ExchCXX::Kernel::PW91_LDA_RPA,   "LDA_C_PW_RPA"      },

    // GGA Functionals
    { ExchCXX::Kernel::PBE_X,          "GGA_X_PBE"         },
    { ExchCXX::Kernel::PBE_C,          "GGA_C_PBE"         },
    { ExchCXX::Kernel::revPBE_X,       "GGA_X_PBE_R"       },
    { ExchCXX::Kernel::B88,            "GGA_X_B88"         },
    { ExchCXX::Kernel::LYP,            "GGA_C_LYP"         },
    { ExchCXX::Kernel::B97_D,          "GGA_XC_B97_D"      },
    { ExchCXX::Kernel::ITYH_X,         "GGA_X_ITYH"        },
    { ExchCXX::Kernel::P86_C,          "GGA_C_P86"         },
    { ExchCXX::Kernel::P86VWN_FT_C,    "GGA_C_P86VWN_FT"   },
    { ExchCXX::Kernel::PW91_C,         "GGA_C_PW91"        },
    { ExchCXX::Kernel::PBE_SOL_C,      "GGA_C_PBE_SOL"     },
    { ExchCXX::Kernel::BMK_C,          "GGA_C_BMK"         },
    { ExchCXX::Kernel::N12_C,          "GGA_C_N12"         },
    { ExchCXX::Kernel::N12_SX_C,       "GGA_C_N12_SX"      },
    { ExchCXX::Kernel::SOGGA11_X_C,    "GGA_C_SOGGA11_X"   },
    { ExchCXX::Kernel::PW91_X,         "GGA_X_PW91"        },
    { ExchCXX::Kernel::MPW91_X,        "GGA_X_MPW91"       },
    { ExchCXX::Kernel::OPTX_X,         "GGA_X_OPTX"        },
    { ExchCXX::Kernel::RPBE_X,         "GGA_X_RPBE"        },
    { ExchCXX::Kernel::SOGGA11_X_X,    "HYB_GGA_X_SOGGA11_X" },
    { ExchCXX::Kernel::PW86_X,         "GGA_X_PW86"        },
    { ExchCXX::Kernel::wB97_XC,        "HYB_GGA_XC_WB97"   },
    { ExchCXX::Kernel::wB97X_XC,       "HYB_GGA_XC_WB97X"  },
    { ExchCXX::Kernel::wB97X_V_XC,     "HYB_GGA_XC_WB97X_V"},
    { ExchCXX::Kernel::wB97X_D_XC,     "HYB_GGA_XC_WB97X_D"},
    { ExchCXX::Kernel::wB97X_D3_XC,    "HYB_GGA_XC_WB97X_D3"},
    { ExchCXX::Kernel::HJS_PBE_X,      "GGA_X_HJS_PBE" },
    { ExchCXX::Kernel::wPBEh_X_default0, "GGA_X_WPBEH" },

    // MGGA Functionals
    { ExchCXX::Kernel::SCAN_C,         "MGGA_C_SCAN"       },
    { ExchCXX::Kernel::SCAN_X,         "MGGA_X_SCAN"       },
    { ExchCXX::Kernel::SCANL_C,        "MGGA_C_SCANL"      },
    { ExchCXX::Kernel::SCANL_X,        "MGGA_X_SCANL"      },
    { ExchCXX::Kernel::R2SCAN_C,       "MGGA_C_R2SCAN"     },
    { ExchCXX::Kernel::R2SCAN_X,       "MGGA_X_R2SCAN"     },
    { ExchCXX::Kernel::R2SCANL_C,      "MGGA_C_R2SCANL"    },
    { ExchCXX::Kernel::R2SCANL_X,      "MGGA_X_R2SCANL"    },
    { ExchCXX::Kernel::FT98_X,         "MGGA_X_FT98"       },
    { ExchCXX::Kernel::M062X_X,        "HYB_MGGA_X_M06_2X" },
    { ExchCXX::Kernel::M062X_C,        "MGGA_C_M06_2X"     },
    { ExchCXX::Kernel::PKZB_X,         "MGGA_X_PKZB"       },
    { ExchCXX::Kernel::PKZB_C,         "MGGA_C_PKZB"       },
    { ExchCXX::Kernel::TPSS_X,         "MGGA_X_TPSS"       },
    { ExchCXX::Kernel::revTPSS_X,      "MGGA_X_REVTPSS"    },
    { ExchCXX::Kernel::M06_L_X,        "MGGA_X_M06_L"      },
    { ExchCXX::Kernel::M06_X,          "HYB_MGGA_X_M06"    },
    { ExchCXX::Kernel::revM06_L_X,     "MGGA_X_REVM06_L"   },
    { ExchCXX::Kernel::M06_HF_X,       "HYB_MGGA_X_M06_HF" },
    { ExchCXX::Kernel::M06_SX_X,       "HYB_MGGA_X_M06_SX" },
    { ExchCXX::Kernel::M06_L_C,        "MGGA_C_M06_L"      },
    { ExchCXX::Kernel::M06_C,          "MGGA_C_M06"        },
    { ExchCXX::Kernel::revM06_L_C,     "MGGA_C_REVM06_L"   },
    { ExchCXX::Kernel::M06_HF_C,       "MGGA_C_M06_HF"     },
    { ExchCXX::Kernel::M06_SX_C,       "MGGA_C_M06_SX"     },
    { ExchCXX::Kernel::M05_2X_C,       "MGGA_C_M05_2X"     },
    { ExchCXX::Kernel::M05_C,          "MGGA_C_M05"        },
    { ExchCXX::Kernel::M08_HX_C,       "MGGA_C_M08_HX"     },
    { ExchCXX::Kernel::M08_SO_C,       "MGGA_C_M08_SO"     },
    { ExchCXX::Kernel::CF22D_C,        "MGGA_C_CF22D"      },
    { ExchCXX::Kernel::M11_C,          "MGGA_C_M11"        },
    { ExchCXX::Kernel::MN12_L_C,       "MGGA_C_MN12_L"     },
    { ExchCXX::Kernel::MN12_SX_C,      "MGGA_C_MN12_SX"    },
    { ExchCXX::Kernel::MN15_C,         "MGGA_C_MN15"       },
    { ExchCXX::Kernel::MN15_L_C,       "MGGA_C_MN15_L"     },
    { ExchCXX::Kernel::TPSS_C,         "MGGA_C_TPSS"       },
    { ExchCXX::Kernel::revTPSS_C,      "MGGA_C_REVTPSS"    },
    { ExchCXX::Kernel::RSCAN_C,        "MGGA_C_RSCAN"      },
    { ExchCXX::Kernel::BC95_C,         "MGGA_C_BC95"       },
    { ExchCXX::Kernel::mBEEF_X,        "MGGA_X_MBEEF"      },
    { ExchCXX::Kernel::RSCAN_X,        "MGGA_X_RSCAN"      },
    { ExchCXX::Kernel::BMK_X,          "HYB_MGGA_X_BMK"    },
    { ExchCXX::Kernel::M08_HX_X,       "HYB_MGGA_X_M08_HX" },
    { ExchCXX::Kernel::M08_SO_X,       "HYB_MGGA_X_M08_SO" },
    { ExchCXX::Kernel::MN12_L_X,       "MGGA_X_MN12_L"     },
    { ExchCXX::Kernel::MN15_L_X,       "MGGA_X_MN15_L"     },
    { ExchCXX::Kernel::MN15_X,         "HYB_MGGA_X_MN15"   },
    { ExchCXX::Kernel::CF22D_X,        "HYB_MGGA_X_CF22D"  },
    { ExchCXX::Kernel::MN12_SX_X,      "HYB_MGGA_X_MN12_SX"},
    { ExchCXX::Kernel::M11_X,          "HYB_MGGA_X_M11"    },
    { ExchCXX::Kernel::M05_X,          "HYB_MGGA_X_M05"    },
    { ExchCXX::Kernel::M05_2X_X,       "HYB_MGGA_X_M05_2X" },

    // KEDFs
    { ExchCXX::Kernel::PC07_K,         "MGGA_K_PC07"       },
    { ExchCXX::Kernel::PC07OPT_K,      "MGGA_K_PC07_OPT"   },
  };

  std::once_flag g_exchcxx_init_once;

  inline void ensure_exchcxx_initialized(ExchCXX::Spin spin) {
    std::call_once(g_exchcxx_init_once, [spin]{ ExchCXX::initialize(spin); });
    //g_exchcxx_users.fetch_add(1, std::memory_order_relaxed);
  }

  inline void maybe_finalize_exchcxx() {
    // Usually safer to never finalize until process exit.
    // If you do want refcounted finalize, uncomment:
    // if(g_exchcxx_users.fetch_sub(1, std::memory_order_relaxed) == 1)
    //   ExchCXX::finalize();
  }

  static bool is_composite_or_hybrid_name(const std::string& s_upper){
    // Heuristics: hybrids or explicit XC combos are not single kernels
    if(s_upper.find("HYB_") != std::string::npos) return true;
    if(s_upper.find("_XC_") != std::string::npos)  return true; // exchange+correlation in one label
    // Also common composites: e.g., "B88+LYP", "PBE0", "SCAN-RVV10", etc.
    if(s_upper.find('+') != std::string::npos)     return true;
    if(s_upper.find("RVV10") != std::string::npos) return true;
    if(s_upper.find("VV10")  != std::string::npos) return true;
    if(s_upper.find("D3")    != std::string::npos) return true;
    if(s_upper.find("D4")    != std::string::npos) return true;
    if(s_upper.find("DISP")  != std::string::npos) return true;
    if(s_upper.find("WB97")  != std::string::npos) return true;
    if(s_upper.find("CAM")   != std::string::npos) return true;
    return false;
  }

  // case-insensitive equality
  static bool iequals(const std::string& a, const std::string& b){
    if(a.size() != b.size()) return false;
    for(size_t i=0;i<a.size();++i){
      if(std::toupper(unsigned(a[i])) != std::toupper(unsigned(b[i]))) return false;
    }
    return true;
  }

  // Invert your libxc_kernel_map: LibXC name -> Kernel (case-insensitive match)
  static std::optional<ExchCXX::Kernel>
  kernel_from_libxc_name(const std::string& libxc_name){
    for(const auto& kv : libxc_kernel_map){
      if(iequals(kv.second, libxc_name)) return kv.first;
    }
    return std::nullopt;
  }

  // True for Laplacian-requiring single-kernel variants (SCANL, R2SCANL, etc.)
  static bool libxc_name_needs_lapl(const std::string& libxc_name_upper){
    return (libxc_name_upper.find("_R2SCANL") != std::string::npos) ||
      (libxc_name_upper.find("_SCANL")   != std::string::npos);
  }


  // Density cutoff value matching Libxc behavior
  // Add before GDFT_xc_gga or as a helper
  static void apply_density_screening_gga(sycl::queue* stream,
                                          int np, const double* rho, bool polarized,
                                          double* eps, double* vrho, double* vsigma,
                                          double* v2rho2, double* v2rhosigma, double* v2sigma2,
                                          int order, double cutoff = 1e-15)  // Match libxc's B3LYP threshold
  {
    stream->parallel_for(sycl::range<1>(np), [=](sycl::id<1> idx) {
      const int i = static_cast<int>(idx[0]);
    
      const double rho_total = polarized 
        ? (rho[2*i] + rho[2*i + 1]) 
        : rho[i];
    
      if (rho_total < cutoff) {
        if (eps) eps[i] = 0.0;
      
        if (order >= 1) {
          if (polarized) {
            vrho[2*i] = 0.0;
            vrho[2*i + 1] = 0.0;
            vsigma[3*i] = 0.0;
            vsigma[3*i + 1] = 0.0;
            vsigma[3*i + 2] = 0.0;
          } else {
            vrho[i] = 0.0;
            vsigma[i] = 0.0;
          }
        }
      
        if (order >= 2) {
          if (polarized) {
            v2rho2[3*i] = 0.0;
            v2rho2[3*i + 1] = 0.0;
            v2rho2[3*i + 2] = 0.0;
            for (int j = 0; j < 6; ++j) {
              v2rhosigma[6*i + j] = 0.0;
              v2sigma2[6*i + j] = 0.0;
            }
          } else {
            v2rho2[i] = 0.0;
            v2rhosigma[i] = 0.0;
            v2sigma2[i] = 0.0;
          }
        }
      }
    }).wait();
  }
  // constexpr double RHO_CUTOFF = 1e-20;
  // // SYCL kernel to apply density screening for GGA functionals
  // // Sets derivatives to zero for grid points with rho < RHO_CUTOFF
  // inline void apply_density_cutoff_gga(
  //   sycl::queue& q,
  //   int np,
  //   const double* rho,    // [nspin*np] or [np] for RKS
  //   int nspin,            // 1 for RKS, 2 for UKS
  //   double* eps,          // [np] - can be nullptr
  //   double* vrho,         // [nspin*np] - can be nullptr
  //   double* vsigma,       // [nv*np] - can be nullptr
  //   double* v2rho2,       // [nv2*np] - can be nullptr
  //   double* v2rhosigma,   // [nv2*np] - can be nullptr
  //   double* v2sigma2      // [nv2*np] - can be nullptr
  // ) {
  //   q.submit([&](sycl::handler& h) {
  //     h.parallel_for(sycl::range<1>(np), [=](sycl::id<1> i) {
  //       // Check if total density is below cutoff
  //       double rho_total = rho[i];
  //       if(nspin == 2) rho_total += rho[np + i];  // UKS: rho_alpha + rho_beta

  //       if(rho_total < RHO_CUTOFF) {
  //         // Zero out all derivatives for this grid point
  //         if(eps) eps[i] = 0.0;
  //         if(vrho) {
  //           for(int s = 0; s < nspin; s++) vrho[s*np + i] = 0.0;
  //         }
  //         if(vsigma) {
  //           int nv = (nspin == 1) ? 1 : 3;
  //           for(int v = 0; v < nv; v++) vsigma[v*np + i] = 0.0;
  //         }
  //         if(v2rho2) {
  //           int nv = (nspin == 1) ? 1 : 3;
  //           for(int v = 0; v < nv; v++) v2rho2[v*np + i] = 0.0;
  //         }
  //         if(v2rhosigma) {
  //           int nv = (nspin == 1) ? 2 : 6;
  //           for(int v = 0; v < nv; v++) v2rhosigma[v*np + i] = 0.0;
  //         }
  //         if(v2sigma2) {
  //           int nv = (nspin == 1) ? 1 : 6;
  //           for(int v = 0; v < nv; v++) v2sigma2[v*np + i] = 0.0;
  //         }
  //       }
  //     });
  //   }).wait();
  // }

}


/* ---------------- Version / reference ---------------- */
static const char* kRef   = "ExchCXX GPU shim (libxc ABI)";
static const char* kDOI   = "";
static const char* kKey   = "ExchCXX";
static const char* kVers  = "ExchCXX-SYCL 1.0";

extern "C" {
const char *xc_reference(void)      { return kRef; }
const char *xc_reference_doi(void)  { return kDOI; }
const char *xc_reference_key(void)  { return kKey; }
void xc_version(int *maj,int *min,int *mic){ if(maj) *maj=1; if(min)*min=0; if(mic)*mic=0; }
const char *xc_version_string(void) { return kVers; }
}


static const std::map<int, std::string> libxc_id_to_name = {
  // --- LDA Exchange ---
  {1,   "LDA_X"},
  {600, "LDA_X_1D_EXPONENTIAL"},
  {21,  "LDA_X_1D_SOFT"},
  {19,  "LDA_X_2D"},
  {546, "LDA_X_ERF"},
  {549, "LDA_X_RAE"},
  {532, "LDA_X_REL"},
  {692, "LDA_X_SLOC"},
  {641, "LDA_X_YUKAWA"},

  // --- LDA Correlation ---
  {18,  "LDA_C_1D_CSC"},
  {26,  "LDA_C_1D_LOOS"},
  {15,  "LDA_C_2D_AMGB"},
  {16,  "LDA_C_2D_PRM"},
  {552, "LDA_C_BR78"},
  {287, "LDA_C_CHACHIYO"},
  {307, "LDA_C_CHACHIYO_MOD"},
  {328, "LDA_C_EPC17"},
  {329, "LDA_C_EPC17_2"},
  {330, "LDA_C_EPC18_1"},
  {331, "LDA_C_EPC18_2"},
  {578, "LDA_C_GK72"},
  {5,   "LDA_C_GL"},
  {24,  "LDA_C_GOMBAS"},
  {4,   "LDA_C_HL"},
  {579, "LDA_C_KARASIEV"},
  {308, "LDA_C_KARASIEV_MOD"},
  {551, "LDA_C_MCWEENY"},
  {22,  "LDA_C_ML1"},
  {23,  "LDA_C_ML2"},
  {14,  "LDA_C_OB_PW"},
  {11,  "LDA_C_OB_PZ"},
  {574, "LDA_C_OW"},
  {573, "LDA_C_OW_LYP"},
  {554, "LDA_C_PK09"},
  {590, "LDA_C_PMGB06"},
  {12,  "LDA_C_PW"},
  {654, "LDA_C_PW_ERF"},
  {13,  "LDA_C_PW_MOD"},
  {25,  "LDA_C_PW_RPA"},
  {9,   "LDA_C_PZ"},
  {10,  "LDA_C_PZ_MOD"},
  {27,  "LDA_C_RC04"},
  {3,   "LDA_C_RPA"},
  {684, "LDA_C_RPW92"},
  {683, "LDA_C_UPW92"},
  {17,  "LDA_C_VBH"},
  {7,   "LDA_C_VWN"},
  {28,  "LDA_C_VWN_1"},
  {29,  "LDA_C_VWN_2"},
  {30,  "LDA_C_VWN_3"},
  {31,  "LDA_C_VWN_4"},
  {8,   "LDA_C_VWN_RPA"},
  {317, "LDA_C_W20"},
  {2,   "LDA_C_WIGNER"},
  {6,   "LDA_C_XALPHA"},

  // --- LDA Exchange–Correlation ---
  {536, "LDA_XC_1D_EHWLRG_1"},
  {537, "LDA_XC_1D_EHWLRG_2"},
  {538, "LDA_XC_1D_EHWLRG_3"},
  {318, "LDA_XC_CORRKSDT"},
  {577, "LDA_XC_GDSMFB"},
  {259, "LDA_XC_KSDT"},
  {547, "LDA_XC_LP_A"},
  {548, "LDA_XC_LP_B"},
  {20,  "LDA_XC_TETER93"},
  {599, "LDA_XC_TIH"},
  {43,  "LDA_XC_ZLP"},

  // --- LDA kinetic ---
  {51,  "LDA_K_LP"},
  {580, "LDA_K_LP96"},
  {50,  "LDA_K_TF"},
  {550, "LDA_K_ZLP"},

  // --- Hybrid LDA exchange ---
  {653, "HYB_LDA_X_ERF"},

  // --- Hybrid LDA exchange-correlation ---
  {588, "HYB_LDA_XC_BN05"},
  {178, "HYB_LDA_XC_CAM_LDA0"},
  {177, "HYB_LDA_XC_LDA0"},

  // --- GGA exchange ---
  {128, "GGA_X_2D_B86"},
  {124, "GGA_X_2D_B86_MGC"},
  {127, "GGA_X_2D_B88"},
  {129, "GGA_X_2D_PBE"},
  {192, "GGA_X_AIRY"},
  {56,  "GGA_X_AK13"},
  {120, "GGA_X_AM05"},
  {184, "GGA_X_APBE"},
  {103, "GGA_X_B86"},
  {105, "GGA_X_B86_MGC"},
  {41,  "GGA_X_B86_R"},
  {106, "GGA_X_B88"},
  {179, "GGA_X_B88_6311G"},
  {570, "GGA_X_B88M"},
  {125, "GGA_X_BAYESIAN"},
  {38,  "GGA_X_BCGP"},
  {285, "GGA_X_BEEFVDW"},
  {338, "GGA_X_BKL1"},
  {339, "GGA_X_BKL2"},
  {98,  "GGA_X_BPCCAC"},
  {158, "GGA_X_C09X"},
  {270, "GGA_X_CAP"},
  {298, "GGA_X_CHACHIYO"},
  {111, "GGA_X_DK87_R1"},
  {112, "GGA_X_DK87_R2"},
  {271, "GGA_X_EB88"},
  {215, "GGA_X_ECMV92"},
  {35,  "GGA_X_EV93"},
  {604, "GGA_X_FD_LB94"},
  {605, "GGA_X_FD_REVLB94"},
  {114, "GGA_X_FT97_A"},
  {115, "GGA_X_FT97_B"},
  {107, "GGA_X_G96"},
  {32,  "GGA_X_GAM"},
  {535, "GGA_X_GG99"},
  {34,  "GGA_X_HCTH_A"},
  {527, "GGA_X_HJS_B88"},
  {46,  "GGA_X_HJS_B88_V2"},
  {528, "GGA_X_HJS_B97X"},
  {525, "GGA_X_HJS_PBE"},
  {526, "GGA_X_HJS_PBE_SOL"},
  {191, "GGA_X_HTBS"},
  {529, "GGA_X_ITYH"},
  {622, "GGA_X_ITYH_OPTX"},
  {623, "GGA_X_ITYH_PBE"},
  {544, "GGA_X_KGG99"},
  {145, "GGA_X_KT1"},
  {193, "GGA_X_LAG"},
  {44,  "GGA_X_LAMBDA_CH_N"},
  {45,  "GGA_X_LAMBDA_LO_N"},
  {40,  "GGA_X_LAMBDA_OC2_N"},
  {160, "GGA_X_LB"},
  {182, "GGA_X_LBM"},
  {113, "GGA_X_LG93"},
  {168, "GGA_X_LSPBE"},
  {169, "GGA_X_LSRPBE"},
  {58,  "GGA_X_LV_RPW86"},
  {149, "GGA_X_MB88"},
  {122, "GGA_X_MPBE"},
  {119, "GGA_X_MPW91"},
  {82,  "GGA_X_N12"},
  {180, "GGA_X_NCAP"},
  {324, "GGA_X_NCAPR"},
  {183, "GGA_X_OL2"},
  {171, "GGA_X_OPTB86B_VDW"},
  {139, "GGA_X_OPTB88_VDW"},
  {141, "GGA_X_OPTPBE_VDW"},
  {110, "GGA_X_OPTX"},
  {101, "GGA_X_PBE"},
  {655, "GGA_X_PBE_ERF_GWS"},
  {321, "GGA_X_PBE_GAUSSIAN"},
  {126, "GGA_X_PBE_JSJR"},
  {320, "GGA_X_PBE_MOD"},
  {49,  "GGA_X_PBE_MOL"},
  {102, "GGA_X_PBE_R"},
  {116, "GGA_X_PBE_SOL"},
  {59,  "GGA_X_PBE_TCA"},
  {121, "GGA_X_PBEA"},
  {265, "GGA_X_PBEFE"},
  {60,  "GGA_X_PBEINT"},
  {140, "GGA_X_PBEK1_VDW"},
  {539, "GGA_X_PBEPOW"},
  {291, "GGA_X_PBETRANS"},
  {108, "GGA_X_PW86"},
  {109, "GGA_X_PW91"},
  {316, "GGA_X_PW91_MOD"},
  {734, "GGA_X_Q1D"},
  {48,  "GGA_X_Q2D"},
  {312, "GGA_X_REVSSB_D"},
  {142, "GGA_X_RGE2"},
  {117, "GGA_X_RPBE"},
  {144, "GGA_X_RPW86"},
  {495, "GGA_X_S12G"},
  {530, "GGA_X_SFAT"},
  {601, "GGA_X_SFAT_PBE"},
  {533, "GGA_X_SG4"},
  {150, "GGA_X_SOGGA"},
  {151, "GGA_X_SOGGA11"},
  {91,  "GGA_X_SSB"},
  {92,  "GGA_X_SSB_D"},
  {90,  "GGA_X_SSB_SW"},
  {68,  "GGA_X_VMT84_GE"},
  {69,  "GGA_X_VMT84_PBE"},
  {70,  "GGA_X_VMT_GE"},
  {71,  "GGA_X_VMT_PBE"},
  {118, "GGA_X_WC"},
  {524, "GGA_X_WPBEH"},
  {123, "GGA_X_XPBE"},

  // --- GGA correlation ---
  {39,  "GGA_C_ACGGA"},
  {176, "GGA_C_ACGGAP"},
  {135, "GGA_C_AM05"},
  {186, "GGA_C_APBE"},
  {280, "GGA_C_BMK"},
  {313, "GGA_C_CCDF"},
  {309, "GGA_C_CHACHIYO"},
  {565, "GGA_C_CS1"},
  {88,  "GGA_C_FT97"},
  {33,  "GGA_C_GAM"},
  {555, "GGA_C_GAPC"},
  {556, "GGA_C_GAPLOC"},
  {97,  "GGA_C_HCTH_A"},
  {283, "GGA_C_HYB_TAU_HCTH"},
  {137, "GGA_C_LM"},
  {131, "GGA_C_LYP"},
  {624, "GGA_C_LYPR"},
  {712, "GGA_C_MGGAC"},
  {80,  "GGA_C_N12"},
  {79,  "GGA_C_N12_SX"},
  {87,  "GGA_C_OP_B88"},
  {85,  "GGA_C_OP_G96"},
  {86,  "GGA_C_OP_PBE"},
  {262, "GGA_C_OP_PW91"},
  {84,  "GGA_C_OP_XALPHA"},
  {200, "GGA_C_OPTC"},
  {132, "GGA_C_P86"},
  {217, "GGA_C_P86_FT"},
  {252, "GGA_C_P86VWN"},
  {253, "GGA_C_P86VWN_FT"},
  {130, "GGA_C_PBE"},
  {657, "GGA_C_PBE_ERF_GWS"},
  {322, "GGA_C_PBE_GAUSSIAN"},
  {138, "GGA_C_PBE_JRGX"},
  {272, "GGA_C_PBE_MOL"},
  {133, "GGA_C_PBE_SOL"},
  {216, "GGA_C_PBE_VWN"},
  {258, "GGA_C_PBEFE"},
  {62,  "GGA_C_PBEINT"},
  {246, "GGA_C_PBELOC"},
  {134, "GGA_C_PW91"},
  {47,  "GGA_C_Q2D"},
  {83,  "GGA_C_REGTPSS"},
  {99,  "GGA_C_REVTCA"},
  {143, "GGA_C_RGE2"},
  {553, "GGA_C_SCAN_E0"},
  {534, "GGA_C_SG4"},
  {152, "GGA_C_SOGGA11"},
  {159, "GGA_C_SOGGA11_X"},
  {89,  "GGA_C_SPBE"},
  {281, "GGA_C_TAU_HCTH"},
  {100, "GGA_C_TCA"},
  {559, "GGA_C_TM_LYP"},
  {560, "GGA_C_TM_PBE"},
  {561, "GGA_C_W94"},
  {148, "GGA_C_WI"},
  {153, "GGA_C_WI0"},
  {147, "GGA_C_WL"},
  {136, "GGA_C_XPBE"},
  {61,  "GGA_C_ZPBEINT"},
  {63,  "GGA_C_ZPBESOL"},
  {557, "GGA_C_ZVPBEINT"},
  {606, "GGA_C_ZVPBELOC"},
  {558, "GGA_C_ZVPBESOL"},

  // --- GGA exchange–correlation ---
  {327, "GGA_XC_B97_3C"},
  {170, "GGA_XC_B97_D"},
  {96,  "GGA_XC_B97_GGA1"},
  {286, "GGA_XC_BEEFVDW"},
  {165, "GGA_XC_EDF1"},
  {162, "GGA_XC_HCTH_120"},
  {163, "GGA_XC_HCTH_147"},
  {164, "GGA_XC_HCTH_407"},
  {93,  "GGA_XC_HCTH_407P"},
  {161, "GGA_XC_HCTH_93"},
  {95,  "GGA_XC_HCTH_P14"},
  {94,  "GGA_XC_HCTH_P76"},
  {545, "GGA_XC_HLE16"},
  {167, "GGA_XC_KT1"},
  {146, "GGA_XC_KT2"},
  {587, "GGA_XC_KT3"},
  {194, "GGA_XC_MOHLYP"},
  {195, "GGA_XC_MOHLYP2"},
  {174, "GGA_XC_MPWLYP1W"},
  {181, "GGA_XC_NCAP"},
  {67,  "GGA_XC_OBLYP_D"},
  {65,  "GGA_XC_OPBE_D"},
  {66,  "GGA_XC_OPWLYP_D"},
  {173, "GGA_XC_PBE1W"},
  {175, "GGA_XC_PBELYP1W"},
  {154, "GGA_XC_TH1"},
  {155, "GGA_XC_TH2"},
  {156, "GGA_XC_TH3"},
  {157, "GGA_XC_TH4"},
  {197, "GGA_XC_TH_FC"},
  {198, "GGA_XC_TH_FCFO"},
  {199, "GGA_XC_TH_FCO"},
  {196, "GGA_XC_TH_FL"},
  {255, "GGA_XC_VV10"},
  {166, "GGA_XC_XLYP"},

  // --- GGA kinetic ---
  {506, "GGA_K_ABSP1"},
  {507, "GGA_K_ABSP2"},
  {277, "GGA_K_ABSP3"},
  {278, "GGA_K_ABSP4"},
  {185, "GGA_K_APBE"},
  {54,  "GGA_K_APBEINT"},
  {504, "GGA_K_BALTIN"},
  {516, "GGA_K_DK"},
  {520, "GGA_K_ERNZERHOF"},
  {597, "GGA_K_EXP4"},
  {514, "GGA_K_FR_B88"},
  {515, "GGA_K_FR_PW86"},
  {591, "GGA_K_GDS08"},
  {501, "GGA_K_GE2"},
  {592, "GGA_K_GHDS10"},
  {593, "GGA_K_GHDS10R"},
  {502, "GGA_K_GOLDEN"},
  {510, "GGA_K_GP85"},
  {508, "GGA_K_GR"},
  {521, "GGA_K_LC94"},
  {620, "GGA_K_LGAP"},
  {633, "GGA_K_LGAP_GE"},
  {505, "GGA_K_LIEB"},
  {613, "GGA_K_LKT"},
  {522, "GGA_K_LLP"},
  {509, "GGA_K_LUDENA"},
  {57,  "GGA_K_MEYER"},
  {512, "GGA_K_OL1"},
  {513, "GGA_K_OL2"},
  {616, "GGA_K_PBE2"},
  {595, "GGA_K_PBE3"},
  {596, "GGA_K_PBE4"},
  {511, "GGA_K_PEARSON"},
  {517, "GGA_K_PERDEW"},
  {219, "GGA_K_PG1"},
  {218, "GGA_K_RATIONAL_P"},
  {55,  "GGA_K_REVAPBE"},
  {53,  "GGA_K_REVAPBEINT"},
  {52,  "GGA_K_TFVW"},
  {635, "GGA_K_TFVW_OPT"},
  {523, "GGA_K_THAKKAR"},
  {594, "GGA_K_TKVLN"},
  {187, "GGA_K_TW1"},
  {188, "GGA_K_TW2"},
  {189, "GGA_K_TW3"},
  {190, "GGA_K_TW4"},
  {519, "GGA_K_VJKS"},
  {518, "GGA_K_VSK"},
  {619, "GGA_K_VT84F"},
  {500, "GGA_K_VW"},
  {503, "GGA_K_YT65"},

  // --- HYB_GGA exchange ---
  {646, "HYB_GGA_X_CAM_S12G"},
  {647, "HYB_GGA_X_CAM_S12H"},
  {81,  "HYB_GGA_X_N12_SX"},
  {656, "HYB_GGA_X_PBE_ERF_GWS"},
  {496, "HYB_GGA_X_S12H"},
  {426, "HYB_GGA_X_SOGGA11_X"},

  // --- HYB_GGA exchange–correlation ---
  {607, "HYB_GGA_XC_APBE0"},
  {409, "HYB_GGA_XC_APF"},
  {416, "HYB_GGA_XC_B1LYP"},
  {417, "HYB_GGA_XC_B1PW91"},
  {412, "HYB_GGA_XC_B1WC"},
  {402, "HYB_GGA_XC_B3LYP"},
  {394, "HYB_GGA_XC_B3LYP3"},
  {475, "HYB_GGA_XC_B3LYP5"},
  {461, "HYB_GGA_XC_B3LYP_MCM1"},
  {462, "HYB_GGA_XC_B3LYP_MCM2"},
  {459, "HYB_GGA_XC_B3LYPS"},
  {403, "HYB_GGA_XC_B3P86"},
  {315, "HYB_GGA_XC_B3P86_NWCHEM"},
  {401, "HYB_GGA_XC_B3PW91"},
  {572, "HYB_GGA_XC_B5050LYP"},
  {407, "HYB_GGA_XC_B97"},
  {408, "HYB_GGA_XC_B97_1"},
  {266, "HYB_GGA_XC_B97_1P"},
  {410, "HYB_GGA_XC_B97_2"},
  {414, "HYB_GGA_XC_B97_3"},
  {413, "HYB_GGA_XC_B97_K"},
  {435, "HYB_GGA_XC_BHANDH"},
  {436, "HYB_GGA_XC_BHANDHLYP"},
  {499, "HYB_GGA_XC_BLYP35"},
  {433, "HYB_GGA_XC_CAM_B3LYP"},
  {395, "HYB_GGA_XC_CAM_O3LYP"},
  {681, "HYB_GGA_XC_CAM_PBEH"},
  {490, "HYB_GGA_XC_CAM_QTP_00"},
  {482, "HYB_GGA_XC_CAM_QTP_01"},
  {491, "HYB_GGA_XC_CAM_QTP_02"},
  {614, "HYB_GGA_XC_CAMH_B3LYP"},
  {470, "HYB_GGA_XC_CAMY_B3LYP"},
  {455, "HYB_GGA_XC_CAMY_BLYP"},
  {682, "HYB_GGA_XC_CAMY_PBEH"},
  {477, "HYB_GGA_XC_CAP0"},
  {390, "HYB_GGA_XC_CASE21"},
  {476, "HYB_GGA_XC_EDF2"},
  {608, "HYB_GGA_XC_HAPBE"},
  {314, "HYB_GGA_XC_HFLYP"},
  {431, "HYB_GGA_XC_HJS_B88"},
  {432, "HYB_GGA_XC_HJS_B97X"},
  {429, "HYB_GGA_XC_HJS_PBE"},
  {430, "HYB_GGA_XC_HJS_PBE_SOL"},
  {472, "HYB_GGA_XC_HPBEINT"},
  {427, "HYB_GGA_XC_HSE03"},
  {428, "HYB_GGA_XC_HSE06"},
  {479, "HYB_GGA_XC_HSE12"},
  {480, "HYB_GGA_XC_HSE12S"},
  {481, "HYB_GGA_XC_HSE_SOL"},
  {485, "HYB_GGA_XC_KMLYP"},
  {589, "HYB_GGA_XC_LB07"},
  {400, "HYB_GGA_XC_LC_BLYP"},
  {625, "HYB_GGA_XC_LC_BLYP_EA"},
  {639, "HYB_GGA_XC_LC_BLYPR"},
  {636, "HYB_GGA_XC_LC_BOP"},
  {637, "HYB_GGA_XC_LC_PBEOP"},
  {492, "HYB_GGA_XC_LC_QTP"},
  {469, "HYB_GGA_XC_LC_VV10"},
  {478, "HYB_GGA_XC_LC_WPBE"},
  {488, "HYB_GGA_XC_LC_WPBE08_WHS"},
  {486, "HYB_GGA_XC_LC_WPBE_WHS"},
  {487, "HYB_GGA_XC_LC_WPBEH_WHS"},
  {489, "HYB_GGA_XC_LC_WPBESOL_WHS"},
  {468, "HYB_GGA_XC_LCY_BLYP"},
  {467, "HYB_GGA_XC_LCY_PBE"},
  {473, "HYB_GGA_XC_LRC_WPBE"},
  {465, "HYB_GGA_XC_LRC_WPBEH"},
  {437, "HYB_GGA_XC_MB3LYP_RC04"},
  {640, "HYB_GGA_XC_MCAM_B3LYP"},
  {405, "HYB_GGA_XC_MPW1K"},
  {483, "HYB_GGA_XC_MPW1LYP"},
  {484, "HYB_GGA_XC_MPW1PBE"},
  {418, "HYB_GGA_XC_MPW1PW"},
  {419, "HYB_GGA_XC_MPW3LYP"},
  {415, "HYB_GGA_XC_MPW3PW"},
  {453, "HYB_GGA_XC_MPWLYP1M"},
  {404, "HYB_GGA_XC_O3LYP"},
  {386, "HYB_GGA_XC_OPB3LYP"},
  {456, "HYB_GGA_XC_PBE0_13"},
  {393, "HYB_GGA_XC_PBE38"},
  {290, "HYB_GGA_XC_PBE50"},
  {392, "HYB_GGA_XC_PBE_2X"},
  {273, "HYB_GGA_XC_PBE_MOL0"},
  {276, "HYB_GGA_XC_PBE_MOLB0"},
  {274, "HYB_GGA_XC_PBE_SOL0"},
  {275, "HYB_GGA_XC_PBEB0"},
  {406, "HYB_GGA_XC_PBEH"},
  {460, "HYB_GGA_XC_QTP17"},
  {610, "HYB_GGA_XC_RCAM_B3LYP"},
  {325, "HYB_GGA_XC_RELPBE0"},
  {454, "HYB_GGA_XC_REVB3LYP"},
  {420, "HYB_GGA_XC_SB98_1A"},
  {421, "HYB_GGA_XC_SB98_1B"},
  {422, "HYB_GGA_XC_SB98_1C"},
  {423, "HYB_GGA_XC_SB98_2A"},
  {424, "HYB_GGA_XC_SB98_2B"},
  {425, "HYB_GGA_XC_SB98_2C"},
  {434, "HYB_GGA_XC_TUNED_CAM_B3LYP"},
  {463, "HYB_GGA_XC_WB97"},
  {464, "HYB_GGA_XC_WB97X"},
  {471, "HYB_GGA_XC_WB97X_D"},
  {399, "HYB_GGA_XC_WB97X_D3"},
  {466, "HYB_GGA_XC_WB97X_V"},
  {611, "HYB_GGA_XC_WC04"},
  {615, "HYB_GGA_XC_WHPBE0"},
  {612, "HYB_GGA_XC_WP04"},
  {411, "HYB_GGA_XC_X3LYP"},

  // --- MGGA exchange ---
  {609, "MGGA_X_2D_JS17"},
  {210, "MGGA_X_2D_PRHG07"},
  {211, "MGGA_X_2D_PRHG07_PRP10"},
  {284, "MGGA_X_B00"},
  {207, "MGGA_X_BJ06"},
  {244, "MGGA_X_BLOC"},
  {206, "MGGA_X_BR89"},
  {214, "MGGA_X_BR89_1"},
  {586, "MGGA_X_BR89_EXPLICIT"},
  {602, "MGGA_X_BR89_EXPLICIT_1"},
  {686, "MGGA_X_EDMGGA"},
  {326, "MGGA_X_EEL"},
  {319, "MGGA_X_FT98"},
  {689, "MGGA_X_GDME_0"},
  {690, "MGGA_X_GDME_KOS"},
  {687, "MGGA_X_GDME_NV"},
  {691, "MGGA_X_GDME_VT"},
  {204, "MGGA_X_GVT4"},
  {575, "MGGA_X_GX"},
  {698, "MGGA_X_HLTA"},
  {256, "MGGA_X_JK"},
  {735, "MGGA_X_KTBM_0"},
  {736, "MGGA_X_KTBM_1"},
  {745, "MGGA_X_KTBM_10"},
  {746, "MGGA_X_KTBM_11"},
  {747, "MGGA_X_KTBM_12"},
  {748, "MGGA_X_KTBM_13"},
  {749, "MGGA_X_KTBM_14"},
  {750, "MGGA_X_KTBM_15"},
  {751, "MGGA_X_KTBM_16"},
  {752, "MGGA_X_KTBM_17"},
  {753, "MGGA_X_KTBM_18"},
  {754, "MGGA_X_KTBM_19"},
  {737, "MGGA_X_KTBM_2"},
  {755, "MGGA_X_KTBM_20"},
  {756, "MGGA_X_KTBM_21"},
  {757, "MGGA_X_KTBM_22"},
  {758, "MGGA_X_KTBM_23"},
  {759, "MGGA_X_KTBM_24"},
  {738, "MGGA_X_KTBM_3"},
  {739, "MGGA_X_KTBM_4"},
  {740, "MGGA_X_KTBM_5"},
  {741, "MGGA_X_KTBM_6"},
  {742, "MGGA_X_KTBM_7"},
  {743, "MGGA_X_KTBM_8"},
  {744, "MGGA_X_KTBM_9"},
  {760, "MGGA_X_KTBM_GAP"},
  {342, "MGGA_X_LAK"},
  {201, "MGGA_X_LTA"},
  {203, "MGGA_X_M06_L"},
  {226, "MGGA_X_M11_L"},
  {249, "MGGA_X_MBEEF"},
  {250, "MGGA_X_MBEEFVDW"},
  {716, "MGGA_X_MBR"},
  {696, "MGGA_X_MBRXC_BG"},
  {697, "MGGA_X_MBRXH_BG"},
  {644, "MGGA_X_MCML"},
  {711, "MGGA_X_MGGAC"},
  {230, "MGGA_X_MK00"},
  {243, "MGGA_X_MK00B"},
  {227, "MGGA_X_MN12_L"},
  {260, "MGGA_X_MN15_L"},
  {245, "MGGA_X_MODTPSS"},
  {221, "MGGA_X_MS0"},
  {222, "MGGA_X_MS1"},
  {223, "MGGA_X_MS2"},
  {228, "MGGA_X_MS2_REV"},
  {300, "MGGA_X_MS2B"},
  {301, "MGGA_X_MS2BS"},
  {765, "MGGA_X_MSB86BL"},
  {761, "MGGA_X_MSPBEL"},
  {763, "MGGA_X_MSRPBEL"},
  {724, "MGGA_X_MTASK"},
  {257, "MGGA_X_MVS"},
  {302, "MGGA_X_MVSB"},
  {303, "MGGA_X_MVSBS"},
  {576, "MGGA_X_PBE_GX"},
  {213, "MGGA_X_PKZB"},
  {497, "MGGA_X_R2SCAN"},
  {645, "MGGA_X_R2SCAN01"},
  {718, "MGGA_X_R2SCANL"},
  {650, "MGGA_X_R4SCAN"},
  {626, "MGGA_X_REGTM"},
  {603, "MGGA_X_REGTPSS"},
  {293, "MGGA_X_REVM06_L"},
  {581, "MGGA_X_REVSCAN"},
  {701, "MGGA_X_REVSCANL"},
  {693, "MGGA_X_REVTM"},
  {212, "MGGA_X_REVTPSS"},
  {688, "MGGA_X_RLDA"},
  {766, "MGGA_X_RMSB86BL"},
  {762, "MGGA_X_RMSPBEL"},
  {764, "MGGA_X_RMSRPBEL"},
  {209, "MGGA_X_RPP09"},
  {648, "MGGA_X_RPPSCAN"},
  {493, "MGGA_X_RSCAN"},
  {299, "MGGA_X_RTPSS"},
  {542, "MGGA_X_SA_TPSS"},
  {263, "MGGA_X_SCAN"},
  {700, "MGGA_X_SCANL"},
  {707, "MGGA_X_TASK"},
  {205, "MGGA_X_TAU_HCTH"},
  {208, "MGGA_X_TB09"},
  {225, "MGGA_X_TH"},
  {685, "MGGA_X_TLDA"},
  {540, "MGGA_X_TM"},
  {202, "MGGA_X_TPSS"},
  {651, "MGGA_X_VCML"},
  {541, "MGGA_X_VT84"},

  // --- MGGA correlation ---
  {571, "MGGA_C_B88"},
  {397, "MGGA_C_B94"},
  {240, "MGGA_C_BC95"},
  {387, "MGGA_C_CC"},
  {388, "MGGA_C_CCALDA"},
  {341, "MGGA_C_CF22D"},
  {72,  "MGGA_C_CS"},
  {37,  "MGGA_C_DLDF"},
  {699, "MGGA_C_HLTAPW"},
  {562, "MGGA_C_KCIS"},
  {638, "MGGA_C_KCISK"},
  {237, "MGGA_C_M05"},
  {238, "MGGA_C_M05_2X"},
  {235, "MGGA_C_M06"},
  {236, "MGGA_C_M06_2X"},
  {234, "MGGA_C_M06_HF"},
  {233, "MGGA_C_M06_L"},
  {311, "MGGA_C_M06_SX"},
  {78,  "MGGA_C_M08_HX"},
  {77,  "MGGA_C_M08_SO"},
  {76,  "MGGA_C_M11"},
  {75,  "MGGA_C_M11_L"},
  {74,  "MGGA_C_MN12_L"},
  {73,  "MGGA_C_MN12_SX"},
  {269, "MGGA_C_MN15"},
  {261, "MGGA_C_MN15_L"},
  {239, "MGGA_C_PKZB"},
  {498, "MGGA_C_R2SCAN"},
  {642, "MGGA_C_R2SCAN01"},
  {719, "MGGA_C_R2SCANL"},
  {306, "MGGA_C_REVM06"},
  {294, "MGGA_C_REVM06_L"},
  {172, "MGGA_C_REVM11"},
  {582, "MGGA_C_REVSCAN"},
  {585, "MGGA_C_REVSCAN_VV10"},
  {694, "MGGA_C_REVTM"},
  {241, "MGGA_C_REVTPSS"},
  {643, "MGGA_C_RMGGAC"},
  {649, "MGGA_C_RPPSCAN"},
  {391, "MGGA_C_RREGTM"},
  {494, "MGGA_C_RSCAN"},
  {267, "MGGA_C_SCAN"},
  {292, "MGGA_C_SCAN_RVV10"},
  {584, "MGGA_C_SCAN_VV10"},
  {702, "MGGA_C_SCANL"},
  {703, "MGGA_C_SCANL_RVV10"},
  {704, "MGGA_C_SCANL_VV10"},
  {251, "MGGA_C_TM"},
  {231, "MGGA_C_TPSS"},
  {323, "MGGA_C_TPSS_GAUSSIAN"},
  {247, "MGGA_C_TPSSLOC"},
  {232, "MGGA_C_VSXC"},

  // --- MGGA exchange-correlation ---
  {254, "MGGA_XC_B97M_V"},
  {229, "MGGA_XC_CC06"},
  {288, "MGGA_XC_HLE17"},
  {564, "MGGA_XC_LP90"},
  {64,  "MGGA_XC_OTPSS_D"},
  {242, "MGGA_XC_TPSSLYP1W"},
  {652, "MGGA_XC_VCML_RVV10"},
  {42,  "MGGA_XC_ZLP"},

  // --- MGGA kinetic ---
  {629, "MGGA_K_CSK1"},
  {630, "MGGA_K_CSK4"},
  {631, "MGGA_K_CSK_LOC1"},
  {632, "MGGA_K_CSK_LOC4"},
  {627, "MGGA_K_GEA2"},
  {628, "MGGA_K_GEA4"},
  {617, "MGGA_K_L04"},
  {618, "MGGA_K_L06"},
  {543, "MGGA_K_PC07"},
  {634, "MGGA_K_PC07_OPT"},
  {220, "MGGA_K_PGSL025"},
  {621, "MGGA_K_RDA"},

  // --- HYB MGGA Exchange / XC ---
  {279, "HYB_MGGA_X_BMK"},
  {340, "HYB_MGGA_X_CF22D"},
  {36,  "HYB_MGGA_X_DLDF"},
  {705, "HYB_MGGA_X_JS18"},
  {438, "HYB_MGGA_X_M05"},
  {439, "HYB_MGGA_X_M05_2X"},
  {449, "HYB_MGGA_X_M06"},
  {450, "HYB_MGGA_X_M06_2X"},
  {444, "HYB_MGGA_X_M06_HF"},
  {310, "HYB_MGGA_X_M06_SX"},
  {295, "HYB_MGGA_X_M08_HX"},
  {296, "HYB_MGGA_X_M08_SO"},
  {297, "HYB_MGGA_X_M11"},
  {248, "HYB_MGGA_X_MN12_SX"},
  {268, "HYB_MGGA_X_MN15"},
  {224, "HYB_MGGA_X_MS2H"},
  {474, "HYB_MGGA_X_MVSH"},
  {706, "HYB_MGGA_X_PJS18"},
  {305, "HYB_MGGA_X_REVM06"},
  {304, "HYB_MGGA_X_REVM11"},
  {583, "HYB_MGGA_X_REVSCAN0"},
  {264, "HYB_MGGA_X_SCAN0"},
  {282, "HYB_MGGA_X_TAU_HCTH"},

  // --- HYB MGGA XC ---
  {563, "HYB_MGGA_XC_B0KCIS"},
  {441, "HYB_MGGA_XC_B86B95"},
  {440, "HYB_MGGA_XC_B88B95"},
  {398, "HYB_MGGA_XC_B94_HYB"},
  {598, "HYB_MGGA_XC_B98"},
  {443, "HYB_MGGA_XC_BB1K"},
  {389, "HYB_MGGA_XC_BR3P86"},
  {695, "HYB_MGGA_XC_EDMGGAH"},
  {658, "HYB_MGGA_XC_GAS22"},
  {720, "HYB_MGGA_XC_LC_TMLYP"},
  {445, "HYB_MGGA_XC_MPW1B95"},
  {566, "HYB_MGGA_XC_MPW1KCIS"},
  {446, "HYB_MGGA_XC_MPWB1K"},
  {567, "HYB_MGGA_XC_MPWKCIS1K"},
  {568, "HYB_MGGA_XC_PBE1KCIS"},
  {451, "HYB_MGGA_XC_PW6B95"},
  {442, "HYB_MGGA_XC_PW86B95"},
  {452, "HYB_MGGA_XC_PWB6K"},
  {660, "HYB_MGGA_XC_R2SCAN0"},
  {661, "HYB_MGGA_XC_R2SCAN50"},
  {659, "HYB_MGGA_XC_R2SCANH"},
  {458, "HYB_MGGA_XC_REVTPSSH"},
  {396, "HYB_MGGA_XC_TPSS0"},
  {569, "HYB_MGGA_XC_TPSS1KCIS"},
  {457, "HYB_MGGA_XC_TPSSH"},
  {531, "HYB_MGGA_XC_WB97M_V"},
  {447, "HYB_MGGA_XC_X1B95"},
  {448, "HYB_MGGA_XC_XB1K"}
};


// ----- shared helpers -----

static ExchCXX::Kernel map_name_to_kernel(const std::string& in,
                                          int* family_out,
                                          bool* needs_lapl_out)
{
  using namespace detail; // for helpers & maps you showed

  auto set_family = [&](const std::string& u){
    int fam = -1;
    // Exact family prefixes
    if(u.rfind("LDA_",  0) == 0)      fam = XC_FAMILY_LDA;
    else if(u.rfind("GGA_", 0) == 0)  fam = XC_FAMILY_GGA;
    else if(u.rfind("MGGA_",0) == 0)  fam = XC_FAMILY_MGGA;
    // Hybrids: map to underlying family
    else if(u.rfind("HYB_GGA_", 0) == 0)  fam = XC_FAMILY_GGA;
    else if(u.rfind("HYB_MGGA_",0) == 0)  fam = XC_FAMILY_MGGA;

    if(family_out) *family_out = fam;
  };

  // 1) Normalize input
  std::string s  = in;
  std::string su = to_upper(s);

  if(su.empty()){
    if(family_out) *family_out = -1;
    if(needs_lapl_out) *needs_lapl_out = false;
    throw std::invalid_argument("Empty functional name/id");
  }

  // 2) If user passed a numeric LibXC id, translate it to a LibXC name
  bool is_numeric = std::all_of(su.begin(), su.end(), [](unsigned char c){ return std::isdigit(c); });
  if(is_numeric){
    int id = 0;
    try { id = std::stoi(su); } catch(...) { /* fall through */ }
    auto it = libxc_id_to_name.find(id);
    if(it != libxc_id_to_name.end()){
      su = to_upper(it->second);
    } else {
      if(family_out) *family_out = -1;
      if(needs_lapl_out) *needs_lapl_out = false;
      std::ostringstream oss;
      oss << "Unknown LibXC id: " << su;
      throw std::invalid_argument(oss.str());
    }
  }

  // 3) Reject obvious composite/hybrid-XC combos (this shim is for single kernels)
  if(is_composite_or_hybrid_name(su)){
    if(family_out) *family_out = -1;
    if(needs_lapl_out) *needs_lapl_out = false;
    std::ostringstream oss;
    oss << "Composite/Hybrid-XC label not supported as single kernel: " << su;
    throw std::invalid_argument(oss.str());
  }

  // 4) Resolve LibXC name -> ExchCXX::Kernel using your libxc_kernel_map
  auto optk = kernel_from_libxc_name(su);
  if(!optk){
    if(family_out) *family_out = -1;
    if(needs_lapl_out) *needs_lapl_out = false;
    std::ostringstream oss;
    oss << "No ExchCXX::Kernel mapping for LibXC name: " << su;
    throw std::invalid_argument(oss.str());
  }

  // 5) Fill outs
  set_family(su);
  if(needs_lapl_out) *needs_lapl_out = libxc_name_needs_lapl(su);

  // Debug breadcrumb (optional)
  std::cout << "[map_name_to_kernel] " << in << " -> " << su
            << " | family=" << (family_out ? *family_out : -999)
            << " | needs_lapl=" << (needs_lapl_out ? *needs_lapl_out : false)
            << " | kernel=" << static_cast<int>(*optk) << std::endl;

  return *optk;
}

// static ExchCXX::Kernel map_name_to_kernel(const std::string& in,
//                                           int* family_out, bool* needs_lapl_out)
// {
//   std::cout << "1. setting from here : " << in << std::endl;

//   std::string s = in;
//   for (auto& c : s) c = ::toupper(c);

//   auto set = [&](int fam, bool lapl, ExchCXX::Kernel k){
//     if (family_out) *family_out = fam;
//     if (needs_lapl_out) *needs_lapl_out = lapl;
//     return k;
//   };

//   // ---- LDA (single kernels) ----
//   if (s == "LDA_X" || s == "LDA_X_SLATER" || s == "SLATER")
//     return set(XC_FAMILY_LDA, false, ExchCXX::Kernel::SlaterExchange);


//   if (s == "LDA_C_VWN") {  return set(XC_FAMILY_LDA, false, ExchCXX::Kernel::VWN); }
//   if (s == "LDA_C_VWN_3")   return set(XC_FAMILY_LDA, false, ExchCXX::Kernel::VWN3);
//   if (s == "LDA_C_VWN_RPA") return set(XC_FAMILY_LDA, false, ExchCXX::Kernel::VWN5);

//   // ---- GGA (single kernels) ----
//   if (s == "GGA_X_PBE") return set(XC_FAMILY_GGA, false, ExchCXX::Kernel::PBE_X);
//   if (s == "GGA_C_PBE") return set(XC_FAMILY_GGA, false, ExchCXX::Kernel::PBE_C);
//   if (s == "GGA_X_B88") return set(XC_FAMILY_GGA, false, ExchCXX::Kernel::B88);
//   if (s == "GGA_C_LYP") return set(XC_FAMILY_GGA, false, ExchCXX::Kernel::LYP);
//   // add more as needed (PW91_X, PW91_C, RPBE_X, ...)

//   // ---- mGGA (single kernels) ----
//   if (s == "MGGA_X_SCAN")      return set(XC_FAMILY_MGGA, false, ExchCXX::Kernel::SCAN_X);
//   if (s == "MGGA_C_SCAN")      return set(XC_FAMILY_MGGA, false, ExchCXX::Kernel::SCAN_C);
//   if (s == "MGGA_X_R2SCANL")   return set(XC_FAMILY_MGGA, true,  ExchCXX::Kernel::R2SCANL_X);
//   if (s == "MGGA_C_R2SCANL")   return set(XC_FAMILY_MGGA, true,  ExchCXX::Kernel::R2SCANL_C);
//   // etc.

//   std::fprintf(stderr, "ExchCXX error: unknown single-kernel name '%s'\n", in.c_str());
//   if (family_out) *family_out = -1;
//   if (needs_lapl_out) *needs_lapl_out = false;
//   throw std::invalid_argument("Unknown single-kernel: " + in);
// }

extern "C" int xc_functional_get_number(const char *name) {
  if(!name) return 0;
  int family=0; bool need_lapl=false;
  auto k = map_name_to_kernel(name, &family, &need_lapl);
  // You can’t encode need_lapl separately in this 16+16 scheme unless you dedicate a bit.
  // If you need it later, recompute from the kernel at init time.
  return (family << 16) | static_cast<int>(k);
}

/* ---------------- Shim state kept in xc_func_type::params ---------------- */
struct ShimImpl {
  ExchCXX::Spin spin{};
  int  family = -1;
  bool needs_lapl = false;

  // exactly one of these will be non-null
  std::unique_ptr<ExchCXX::XCKernel>     k;
  std::unique_ptr<ExchCXX::XCFunctional> f;
  bool is_functional() const noexcept { return bool(f); }
};

static inline ShimImpl* get_impl(const xc_func_type* p){
  return reinterpret_cast<ShimImpl*>(p ? p->params : nullptr);
}
template<class Fn>
static int with_xc(const xc_func_type* f, Fn&& fn) {
  auto* impl = get_impl(f);
  if(!impl) return 1;
  if(impl->f) { fn(*impl->f); return 0; }
  if(impl->k) { fn(*impl->k); return 0; }
  return 1; // nothing to call
}
static inline int bad_args(...) { return 1; }


/* Populate minimal dimensions for the arrays we actually use. */
// static void fill_dimensions(xc_dimensions* d, int family, int nspin, bool needs_lapl) {
//   std::memset(d, 0, sizeof(*d));

//   const bool unpol   = (nspin == XC_UNPOLARIZED);
//   const int  rho_dim = unpol ? 1 : 2;
//   const int  sig_dim = unpol ? 1 : 3;   // (aa,ab,bb) for pol
//   const int  lap_dim = rho_dim;
//   const int  tau_dim = rho_dim;

//   // 0th/1st order
//   d->rho   = rho_dim;
//   d->sigma = sig_dim;
//   d->lapl  = lap_dim;
//   d->tau   = tau_dim;

//   d->zk    = 1;
//   d->vrho  = rho_dim;
//   if (family >= XC_FAMILY_GGA)  d->vsigma = sig_dim;
//   if (family >= XC_FAMILY_MGGA) {
//     d->vtau  = tau_dim;
//     d->vlapl = needs_lapl ? lap_dim : 0;
//   }

//   // 2nd order — this is what was missing
//   // LDA
//   d->v2rho2 = unpol ? 1 : 3;  // (aa,ab,bb) for pol

//   // GGA
//   if (family >= XC_FAMILY_GGA) {
//     // rho–sigma mixed block: unpol: 1; pol: 2*rho_channels * 3*sigma_channels = 6
//     d->v2rhosigma = unpol ? 1 : 6;

//     // sigma–sigma block: unpol: 1; pol: symmetric 3x3 => 6 (aa-aa, aa-ab, aa-bb, ab-ab, ab-bb, bb-bb)
//     d->v2sigma2   = unpol ? 1 : 6;
//   }

//   // mGGA
//   if (family >= XC_FAMILY_MGGA) {
//     d->v2rholapl   = needs_lapl ? lap_dim : 0;
//     d->v2rhotau    = rho_dim;
//     d->v2sigmalapl = needs_lapl ? sig_dim : 0;
//     d->v2sigmatau  = sig_dim;
//     d->v2lapl2     = needs_lapl ? lap_dim : 0;
//     d->v2lapltau   = needs_lapl ? lap_dim : 0;
//     d->v2tau2      = rho_dim;
//   }
// }

static void fill_dimensions(xc_dimensions* d, int family, int nspin, bool needs_lapl) {
  std::memset(d, 0, sizeof(*d));

  const bool unpol   = (nspin == XC_UNPOLARIZED);
  const int  rho_dim = unpol ? 1 : 2;
  const int  sig_dim = unpol ? 1 : 3;   // (aa, ab, bb) for pol
  const int  lap_dim = rho_dim;
  const int  tau_dim = rho_dim;

  // Inputs
  d->rho   = rho_dim;
  d->sigma = sig_dim;
  d->lapl  = lap_dim;
  d->tau   = tau_dim;

  // 0th order
  d->zk    = 1;

  // 1st order
  d->vrho  = rho_dim;
  if (family >= XC_FAMILY_GGA)  d->vsigma = sig_dim;
  if (family >= XC_FAMILY_MGGA) {
    d->vtau  = tau_dim;
    d->vlapl = needs_lapl ? lap_dim : 0;
  }

  // 2nd order - LDA
  d->v2rho2 = unpol ? 1 : 3;  // symmetric

  // 2nd order - GGA
  if (family >= XC_FAMILY_GGA) {
    d->v2rhosigma = rho_dim * sig_dim;    // 2*3=6 for pol
    d->v2sigma2   = unpol ? 1 : 6;        // symmetric 3x3
  }

  // 2nd order - mGGA
  if (family >= XC_FAMILY_MGGA) {
    d->v2rholapl   = needs_lapl ? (rho_dim * lap_dim) : 0;  // 2*2=4
    d->v2rhotau    = rho_dim * tau_dim;                      // 2*2=4
    d->v2sigmalapl = needs_lapl ? (sig_dim * lap_dim) : 0;  // 3*2=6
    d->v2sigmatau  = sig_dim * tau_dim;                      // 3*2=6
    d->v2lapl2     = needs_lapl ? (unpol ? 1 : 3) : 0;      // symmetric
    d->v2lapltau   = needs_lapl ? (lap_dim * tau_dim) : 0;  // 2*2=4
    d->v2tau2      = unpol ? 1 : 3;                          // symmetric
  }
}

extern "C" xc_func_type *xc_func_alloc(void) {
  return (xc_func_type*) std::calloc(1, sizeof(xc_func_type));
}

extern "C" int xc_func_init(xc_func_type *p, int functional, int nspin) {
  using detail::to_upper;
  using detail::libxc_name_needs_lapl;
  using detail::kernel_from_libxc_name;
  using detail::libxc_name_to_functional;

  std::cout << "xc_func_init: " << functional << ", " << nspin << std::endl;
  if (!p) return 1;
  if (nspin != XC_UNPOLARIZED && nspin != XC_POLARIZED) return 2;

  auto impl = std::make_unique<ShimImpl>();
  impl->spin = (nspin == XC_UNPOLARIZED)
                 ? ExchCXX::Spin::Unpolarized
                 : ExchCXX::Spin::Polarized;
  std::fprintf(stderr, "[DEBUG] GDFT_xc_gga: nspin=%d, spin=%d\n",
               (impl->spin == ExchCXX::Spin::Polarized) ? 2 : 1,
               int(impl->spin));

  // Detect our packed form: (family<<16 | kernel_enum)
  const int hi = (functional >> 16) & 0xFFFF;
  const int lo =  functional        & 0xFFFF;
  const bool looks_packed =
      (hi == XC_FAMILY_LDA || hi == XC_FAMILY_GGA || hi == XC_FAMILY_MGGA);

  // If not packed, translate LibXC id -> name once
  std::string libxc_name;
  std::string name_upper;

  enum class Path { KernelPacked, FunctionalByName, KernelByName } path = Path::KernelByName;

  if (functional != 0 && looks_packed) {
    path = Path::KernelPacked;
  } else {
    auto it_id = libxc_id_to_name.find(functional);
    if (it_id == libxc_id_to_name.end()) {
      std::fprintf(stderr, "ExchCXX: unknown LibXC ID %d\n", functional);
      return 3; // let caller fallback
    }
    libxc_name = it_id->second;
    name_upper = to_upper(libxc_name);

    // If this label is a composite/hybrid XC functional, use XCFunctional path
    auto fopt = libxc_name_to_functional(name_upper);
    if (fopt) {
      path = Path::FunctionalByName;
    } else {
      path = Path::KernelByName;
    }
  }

  // Initialize ExchCXX device layer (spin-aware) before building kernels/functionals
  detail::ensure_exchcxx_initialized(impl->spin);

  auto finalize = [&](int family, bool needs_lapl) {
    impl->family     = family;
    impl->needs_lapl = needs_lapl;
    p->nspin         = nspin;
    p->params        = impl.release();
    p->params_size   = sizeof(ShimImpl);
    fill_dimensions(&p->dim, family, nspin, needs_lapl);
    return 0;
  };

  try {
    bool needs_lapl = false;
    int  family     = XC_FAMILY_GGA; // corrected below

    // // ---- Custom hybrid override: B3LYP family ----
    // // Handle HYB_GGA_XC_B3LYP and its variants explicitly to control VWN flavor
    // if (name_upper == "HYB_GGA_XC_B3LYP" || name_upper == "HYB_GGA_XC_B3LYP3" || name_upper == "HYB_GGA_XC_B3LYP5") {
    //   auto make_b3lyp_with = [&](ExchCXX::Kernel vwn_kernel) {
    //     ExchCXX::Backend backend = ExchCXX::Backend::builtin;
    //     ExchCXX::Spin spin = impl->spin;

    //     const ExchCXX::HybCoeffs hyb_coefs = {0.20, 0.0, 0.0};
    //     std::vector<std::pair<double, ExchCXX::XCKernel>> terms = {
    //       {0.08, ExchCXX::XCKernel(backend, ExchCXX::Kernel::SlaterExchange, spin)},
    //       {0.72, ExchCXX::XCKernel(backend, ExchCXX::Kernel::B88,           spin)},
    //       {0.19, ExchCXX::XCKernel(backend, vwn_kernel,                     spin)},
    //       {0.81, ExchCXX::XCKernel(backend, ExchCXX::Kernel::LYP,           spin)}
    //     };
    //     return std::make_unique<ExchCXX::XCFunctional>(terms, hyb_coefs);
    //   };

    //   // Map label → kernel flavor
    //   if (name_upper == "HYB_GGA_XC_B3LYP3") {
    //     std::cout << "exchcxx: kernel == VWN3 \n";
    //     impl->f = make_b3lyp_with(ExchCXX::Kernel::VWN3);
    //   } else {
    //     // PySCF ≥ 2.3 default: VWN-RPA = VWN5 in this ExchCXX
    //     // LibXC's HYB_GGA_XC_B3LYP uses LDA_C_VWN_RPA (ID 8) = ExchCXX::Kernel::VWN5
    //     std::cout << "exchcxx: kernel == VWN5 \n";
    //     impl->f = make_b3lyp_with(ExchCXX::Kernel::VWN5);
    //   }

    //   needs_lapl  = false;
    //   family      = XC_FAMILY_GGA;
    //   std::fprintf(stderr, "[gdft] Built XCFunctional (custom B3LYP) for '%s' (VWN5==RPA)\n",
    //                name_upper.c_str());

    //   return finalize(XC_FAMILY_GGA, needs_lapl);
    // }


    if (path == Path::KernelPacked) {
      // Build from packed kernel enum
      const auto kenum = static_cast<ExchCXX::Kernel>(lo);

      // Heuristic laplacian flag from LibXC name if we can map back
      auto it = detail::libxc_kernel_map.find(kenum);
      if (it != detail::libxc_kernel_map.end())
        needs_lapl = libxc_name_needs_lapl(to_upper(it->second));

      impl->k = std::make_unique<ExchCXX::XCKernel>(ExchCXX::Backend::builtin, kenum, impl->spin);

      family = impl->k->is_mgga() ? XC_FAMILY_MGGA
             : impl->k->is_gga()  ? XC_FAMILY_GGA
             :                      XC_FAMILY_LDA;

      std::fprintf(stderr, "[gdft] 1. Built XCKernel: enum=%d family=%d spin=%d\n",
                   int(kenum), family, int(impl->spin));
      std::fprintf(stderr, "[gdft] 1. is_lda=%d is_gga=%d is_mgga=%d\n",
                   impl->k->is_lda(), impl->k->is_gga(), impl->k->is_mgga());

    } else if (path == Path::FunctionalByName) {
        // Build full XC functional (handles hybrids/composites like B3LYP, PBE, SCAN, …)
        const auto fun_opt = libxc_name_to_functional(name_upper);
        if (!fun_opt) {
          std::fprintf(stderr, "ExchCXX: LibXC label '%s' not recognized as a composite functional\n",
                       name_upper.c_str());
          return 3;
        }
        const auto fun = *fun_opt;
        needs_lapl = libxc_name_needs_lapl(name_upper);

        impl->f = std::make_unique<ExchCXX::XCFunctional>(ExchCXX::Backend::builtin, fun, impl->spin);

        family = impl->f->is_mgga() ? XC_FAMILY_MGGA
          : impl->f->is_gga()  ? XC_FAMILY_GGA
          :                      XC_FAMILY_LDA;

        std::fprintf(stderr, "[gdft] 2. Built XCFunctional: '%s' family=%d spin=%d\n",
                     name_upper.c_str(), family, int(impl->spin));
        std::fprintf(stderr, "[gdft] 2. f.is_lda=%d f.is_gga=%d f.is_mgga=%d\n",
                     impl->f->is_lda(), impl->f->is_gga(), impl->f->is_mgga());

    } else { // KernelByName: single kernel by LibXC name
      auto maybe_k = kernel_from_libxc_name(name_upper);
      if (!maybe_k) {
        std::fprintf(stderr,
          "ExchCXX: LibXC name '%s' has no builtin single-kernel implementation\n",
          name_upper.c_str());
        return 3; // let caller fallback
      }
      const auto kenum = *maybe_k;
      needs_lapl = libxc_name_needs_lapl(name_upper);

      impl->k = std::make_unique<ExchCXX::XCKernel>(
                  ExchCXX::Backend::builtin, kenum, impl->spin);

      family = impl->k->is_mgga() ? XC_FAMILY_MGGA
             : impl->k->is_gga()  ? XC_FAMILY_GGA
             :                      XC_FAMILY_LDA;

      std::fprintf(stderr, "[gdft] 3. Built XCKernel: name='%s' enum=%d family=%d spin=%d\n",
                   name_upper.c_str(), int(kenum), family, int(impl->spin));
      std::fprintf(stderr, "[gdft] 3. is_lda=%d is_gga=%d is_mgga=%d\n",
                   impl->k->is_lda(), impl->k->is_gga(), impl->k->is_mgga());
    }

    // Stash and finalize libxc-style handle
    return finalize(family, needs_lapl);

  } catch (const std::exception& e) {
    std::fprintf(stderr, "ExchCXX functional construction failed: %s\n", e.what());
    return 3; // signal caller to fallback
  }
}
// extern "C" int xc_func_init(xc_func_type *p, int functional, int nspin) {
//   using detail::to_upper;
//   using detail::kernel_from_libxc_name;
//   using detail::libxc_name_needs_lapl;

//   std::cout << "xc_func_init: " << functional << std::endl;
//   if (!p) return 1;
//   if (nspin != XC_UNPOLARIZED && nspin != XC_POLARIZED) return 2;

//   auto impl = std::make_unique<ShimImpl>();
//   impl->spin = (nspin == XC_UNPOLARIZED)
//                  ? ExchCXX::Spin::Unpolarized
//                  : ExchCXX::Spin::Polarized;

//   // Detect packed (family<<16 | kernel_enum)
//   const int hi = (functional >> 16) & 0xFFFF;
//   const int lo =  functional        & 0xFFFF;
//   const bool looks_packed =
//       (hi == XC_FAMILY_LDA || hi == XC_FAMILY_GGA || hi == XC_FAMILY_MGGA);

//   // Resolve LibXC id -> name if not packed
//   std::string libxc_name, name_upper;
//   if (!looks_packed) {
//     auto it_id = libxc_id_to_name.find(functional);
//     if (it_id == libxc_id_to_name.end()) {
//       std::fprintf(stderr, "ExchCXX: unknown LibXC ID %d\n", functional);
//       return 3; // let caller fallback
//     }
//     libxc_name = it_id->second;
//     name_upper = to_upper(libxc_name);
//   }

//   // Spin-aware device init (idempotent)
//   detail::ensure_exchcxx_initialized(impl->spin);

//   try {
//     bool needs_lapl = false;
//     int  family     = XC_FAMILY_GGA; // will correct below

//     if (looks_packed) {
//       // -------- Path 1: packed kernel --------
//       const auto kenum = static_cast<ExchCXX::Kernel>(lo);

//       // best-effort laplacian flag via forward map name
//       if (auto it = detail::libxc_kernel_map.find(kenum);
//           it != detail::libxc_kernel_map.end())
//         needs_lapl = libxc_name_needs_lapl(to_upper(it->second));

//       impl->k = std::make_unique<ExchCXX::XCKernel>(
//                   ExchCXX::Backend::builtin, kenum, impl->spin);

//       family = impl->k->is_mgga() ? XC_FAMILY_MGGA
//              : impl->k->is_gga()  ? XC_FAMILY_GGA
//              :                      XC_FAMILY_LDA;

//       std::fprintf(stderr, "[gdft] Built XCKernel: enum=%d family=%d spin=%d\n",
//                    int(kenum), family, int(impl->spin));
//       std::fprintf(stderr, "[gdft] is_lda=%d is_gga=%d is_mgga=%d\n",
//                    impl->k->is_lda(), impl->k->is_gga(), impl->k->is_mgga());

//     } else {
//       // -------- Path 2: try composite/hybrid functional by name --------
//       if (auto fopt = detail::functional_from_libxc_name(libxc_name)) {
//         needs_lapl = libxc_name_needs_lapl(name_upper);

//         impl->f = std::make_unique<ExchCXX::XCFunctional>(
//                     ExchCXX::Backend::builtin, *fopt, impl->spin);

//         family = impl->f->is_mgga() ? XC_FAMILY_MGGA
//                : impl->f->is_gga()  ? XC_FAMILY_GGA
//                :                      XC_FAMILY_LDA;

//         std::fprintf(stderr, "[gdft] Built XCFunctional: '%s' family=%d spin=%d\n",
//                      name_upper.c_str(), family, int(impl->spin));
//         std::fprintf(stderr, "[gdft] f.is_lda=%d f.is_gga=%d f.is_mgga=%d\n",
//                      impl->f->is_lda(), impl->f->is_gga(), impl->f->is_mgga());

//       } else {
//         // -------- Path 3: single kernel by LibXC name --------
//         auto maybe_k = detail::kernel_from_libxc_name(name_upper);
//         if (!maybe_k) {
//           std::fprintf(stderr,
//             "ExchCXX: LibXC name '%s' has no builtin single-kernel implementation\n",
//             name_upper.c_str());
//           return 3; // let caller fallback (e.g., unsupported functional)
//         }
//         const auto kenum = *maybe_k;
//         needs_lapl = libxc_name_needs_lapl(name_upper);

//         impl->k = std::make_unique<ExchCXX::XCKernel>(
//                     ExchCXX::Backend::builtin, kenum, impl->spin);

//         family = impl->k->is_mgga() ? XC_FAMILY_MGGA
//                : impl->k->is_gga()  ? XC_FAMILY_GGA
//                :                      XC_FAMILY_LDA;

//         std::fprintf(stderr,
//                      "[gdft] Built XCKernel: name='%s' enum=%d family=%d spin=%d\n",
//                      name_upper.c_str(), int(kenum), family, int(impl->spin));
//         std::fprintf(stderr, "[gdft] is_lda=%d is_gga=%d is_mgga=%d\n",
//                      impl->k->is_lda(), impl->k->is_gga(), impl->k->is_mgga());
//       }
//     }

//     // Stash and finalize libxc-style handle
//     impl->family     = family;
//     impl->needs_lapl = needs_lapl;

//     p->nspin       = nspin;
//     p->params      = impl.release();
//     p->params_size = sizeof(ShimImpl);

//     fill_dimensions(&p->dim, family, nspin, needs_lapl);
//     return 0;

//   } catch (const std::exception& e) {
//     std::fprintf(stderr, "ExchCXX functional construction failed: %s\n", e.what());
//     return 3; // signal caller to fallback
//   }
// }

extern "C" void xc_func_end(xc_func_type *p) {
  if(!p) return;
  auto *impl = get_impl(p);
  if(impl) {
    delete impl;
    p->params = nullptr;
  }
}

extern "C" void xc_func_free(xc_func_type *p) {
  // if(!p) return;
  // xc_func_end(p);
  // std::free(p);
}

/* ---------------- helpers to detect order from out-structs ---------------- */
// static inline int detect_order_lda (const xc_lda_out_params* out){
//   if(out->v4rho4) return 4;
//   if(out->v3rho3) return 3;
//   if(out->v2rho2) return 2;
//   if(out->vrho)   return 1;
//   if(out->zk)     return 0;
//   return -1;
// }
// static inline int detect_order_gga(const xc_gga_out_params* out) {
//   if(!out) return -1;
//   if(out->v3rho3 || out->v3rho2sigma || out->v3rhosigma2 || out->v3sigma3) return 3;
//   if(out->v2rho2 || out->v2rhosigma || out->v2sigma2) return 2;
//   if(out->vrho   || out->vsigma)                     return 1;
//   if(out->zk)                                        return 0;
//   return -1;
// }
// static inline int detect_order_mgga(const xc_mgga_out_params* out){
//   if(out->v4tau4) return 4;
//   if(out->v3tau3) return 3;
//   if(out->v2tau2) return 2;
//   if(out->vtau || out->vlapl || out->vsigma || out->vrho) return 1;
//   if(out->zk) return 0;
//   return -1;
// }

template<typename T>
static inline int detect_order(const T* out) {
    int order = -1;
    if (out->zk != nullptr) order = 0;
    if (out->vrho != nullptr) order = 1;
    if (out->v2rho2 != nullptr) order = 2;
    if (out->v3rho3 != nullptr) order = 3;
    if (out->v4rho4 != nullptr) order = 4;
    return order;
}

static inline void zero_gga_out(
    sycl::queue& q,
    const xc_func_type* func,
    const xc_gga_out_params* out,
    std::size_t np, int order
) {
  if(order >= 0) q.memset(out->zk, 0, sizeof(double)*np*func->dim.zk);
  if(order >= 1) {
    q.memset(out->vrho, 0, sizeof(double)*np*func->dim.vrho);
    q.memset(out->vsigma, 0, sizeof(double)*np*func->dim.vsigma); // (sigma, lapl, tau)
  }
  if(order >= 2) {
    q.memset(out->v2rho2, 0, sizeof(double)*np*func->dim.v2rho2);
    q.memset(out->v2rhosigma, 0, sizeof(double)*np*func->dim.v2rhosigma);
    q.memset(out->v2sigma2, 0, sizeof(double)*np*func->dim.v2sigma2);
  }
  if(order >= 3) {
    q.memset(out->v3rho3,       0, sizeof(double)*np*func->dim.v3rho3);
    q.memset(out->v3rho2sigma,  0, sizeof(double)*np*func->dim.v3rho2sigma);
    q.memset(out->v3rhosigma2,  0, sizeof(double)*np*func->dim.v3rhosigma2);
    q.memset(out->v3sigma3,     0, sizeof(double)*np*func->dim.v3sigma3);
  }
  if(order >= 4) {
    q.memset(out->v4rho4,       0, sizeof(double)*np*func->dim.v4rho4);
    q.memset(out->v4rho3sigma,  0, sizeof(double)*np*func->dim.v4rho3sigma);
    q.memset(out->v4rho2sigma2, 0, sizeof(double)*np*func->dim.v4rho2sigma2);
    q.memset(out->v4rhosigma3,  0, sizeof(double)*np*func->dim.v4rhosigma3);
    q.memset(out->v4sigma4,     0, sizeof(double)*np*func->dim.v4sigma4);
  }
  q.wait();
}

/* ---------------- Device entry points (AoS on device) ---------------- */
extern "C" int GDFT_xc_lda(
  void* stream_v,
  const xc_func_type *func, int np, const double *rho,
  xc_lda_out_params *out, xc_lda_out_params* /*buf*/
){
  if(!func || !rho || !out || np <= 0) return bad_args();

  const int order = detect_order(out);
  if(order < 0) return 0;
  if(order > 2){
    std::fprintf(stderr, "ExchCXX device: LDA order %d not implemented\n", order);
    return 2;
  }

  auto* stream = reinterpret_cast<sycl::queue*>(stream_v);
  double* eps    = out->zk;
  double* vrho   = out->vrho;
  double* v2rho2 = out->v2rho2;

  // 1) Derivatives first (some backends also write eps here; that’s fine, we’ll overwrite later)
  if(order >= 1){
    int err = with_xc(func, [&](auto& xc){
      xc.eval_exc_vxc_device(np, rho, eps, vrho, stream);
    });
    if(err) return err;
  }
  if(order >= 2){
    int err = with_xc(func, [&](auto& xc){
      xc.eval_vxc_fxc_device(np, rho, vrho, v2rho2, stream);
    });
    if(err) return err;
  }

  // 2) Energy last — this is the authoritative value Python expects to match CPU
  if(eps){
    int err = with_xc(func, [&](auto& xc){
      xc.eval_exc_device(np, rho, eps, stream);
    });
    if(err) return err;
  }

  return 0;
}

// extern "C" int GDFT_xc_lda(
//   void* stream_v,
//   const xc_func_type *func, int np, const double *rho,
//   xc_lda_out_params *out, xc_lda_out_params* /*buf*/
// ){
//   if(!func || !rho || !out || np <= 0) return bad_args();

//   const int order = detect_order_lda(out);
//   if(order < 0) return 0;
//   if(order > 2){
//     std::fprintf(stderr, "ExchCXX device: LDA order %d not implemented\n", order);
//     return 2;
//   }

//   auto* stream = reinterpret_cast<sycl::queue*>(stream_v);
//   double* eps    = out->zk;
//   double* vrho   = out->vrho;
//   double* v2rho2 = out->v2rho2;

//   // Always fill energy first (Python expects zk for all deriv orders).
//   if(eps){
//     int err = with_xc(func, [&](auto& xc){
//       xc.eval_exc_device(np, rho, eps, stream);
//     });
//     if(err) return err;
//   }

//   // Then fill derivatives as requested.
//   if(order >= 1){
//     int err = with_xc(func, [&](auto& xc){
//       xc.eval_exc_vxc_device(np, rho, eps, vrho, stream);
//       // Some backends ignore eps in the *_vxc_* call; we already primed it.
//     });
//     if(err) return err;
//   }
//   if(order >= 2){
//     int err = with_xc(func, [&](auto& xc){
//       xc.eval_vxc_fxc_device(np, rho, vrho, v2rho2, stream);
//       // No energy here either; kept from the priming call.
//     });
//     if(err) return err;
//   }

//   return 0;
// }

// static inline void axpy_field(sycl::queue& q, double* y, const double* x,
//                               double alpha, std::size_t np, int dimc) {
//   if(!y || !x || dimc<=0) return;
//   const std::size_t n = np * std::size_t(dimc);
//   // simple 1D axpy on device
//   q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){
//     y[i] += alpha * x[i];
//   });
// }

// static inline void axpy_gga_out(sycl::queue& q, const xc_dimensions& dim,
//                                 xc_gga_out_params* y,
//                                 const xc_gga_out_params* x,
//                                 double alpha, int order, std::size_t np) {
//   if(order >= 0) axpy_field(q, y->zk, x->zk, alpha, np, dim.zk);
//   if(order >= 1){
//     axpy_field(q, y->vrho,   x->vrho,   alpha, np, dim.vrho);
//     axpy_field(q, y->vsigma, x->vsigma, alpha, np, dim.vsigma);
//   }
//   if(order >= 2){
//     axpy_field(q, y->v2rho2,     x->v2rho2,     alpha, np, dim.v2rho2);
//     axpy_field(q, y->v2rhosigma, x->v2rhosigma, alpha, np, dim.v2rhosigma);
//     axpy_field(q, y->v2sigma2,   x->v2sigma2,   alpha, np, dim.v2sigma2);
//   }
//   //q.wait(); // ensure accumulation is done before next aux
//   // If you ever enable order >= 3 here, add the v3* fields like above.
// }

// extern "C" int GDFT_xc_gga(
//   void* stream_v,
//   const xc_func_type *func, int np,
//   const double *rho, const double *sigma,
//   xc_gga_out_params *out, xc_gga_out_params *buf /* workspace for mix */
// ){
//   if(!func || !rho || !sigma || !out || np <= 0) return bad_args();

//   const int order = detect_order(out);
//   if(order < 0) return 0;
//   if(order > 2){
//     std::fprintf(stderr, "ExchCXX device: GGA order %d not implemented\n", order);
//     return 2;
//   }

//   auto* qptr = reinterpret_cast<sycl::queue*>(stream_v);
//   auto&  q   = *qptr;
//   const auto& dim = func->dim;

//   // ---------- Direct (non-mixed) path ----------
//   if(func->info && func->info->gga && !func->mix_coef){
//     // Zero outputs with correct sizes
//     zero_gga_out(q, func, out, np, order);

//     // Derivatives first (these may write eps too; we’ll overwrite eps after)
//     if(order >= 1){
//       int err = with_xc(func, [&](auto& xc){
//         return xc.eval_exc_vxc_device(np, rho, sigma,
//                                       /*eps=*/out->zk,
//                                       out->vrho, out->vsigma, qptr);
//       });
//       if(err) return err;
//     }
//     if(order >= 2){
//       int err = with_xc(func, [&](auto& xc){
//         return xc.eval_vxc_fxc_device(np, rho, sigma,
//                                       out->vrho, out->vsigma,
//                                       out->v2rho2, out->v2rhosigma, out->v2sigma2,
//                                       qptr);
//       });
//       if(err) return err;
//     }
//     // Energy last — authoritative
//     if(out->zk){
//       int err = with_xc(func, [&](auto& xc){
//         return xc.eval_exc_device(np, rho, sigma, out->zk, qptr);
//       });
//       if(err) return err;
//     }
//     q.wait(); // ensure all kernels complete before returning
//     return 0;
//   }

//   // ---------- Mixed / hybrid path (e.g., B3LYP) ----------
//   if(!func->mix_coef){
//     // Defensive: libxc-like mixes should have mix_coef; if not, nothing to do
//     // (CUDA code returns ierr=0 here)
//     return 0;
//   }

//   if(!buf){
//     std::fprintf(stderr,
//       "ExchCXX device: GGA mix path requires 'buf' workspace (np=%d). "
//       "Caller must provide a device-resident scratch buffer.\n", np);
//     return 2;
//   }

//   // 1) Zero the final accumulator
//   zero_gga_out(q, func, out, np, order);

//   // 2) Loop over components, compute into buf, then out += coef * buf
//   for(int i = 0; i < func->n_func_aux; ++i){
//     const double coef = func->mix_coef[i];
//     const xc_func_type* aux = func->func_aux[i];

//     // Stage: clear buf for this component
//     zero_gga_out(q, func, buf, np, order);

//     // Evaluate this component into buf
//     if(order >= 1){
//       int err = with_xc(aux, [&](auto& xc){
//         return xc.eval_exc_vxc_device(np, rho, sigma,
//                                       /*eps=*/buf->zk,
//                                       buf->vrho, buf->vsigma, qptr);
//       });
//       if(err) return err;
//     }
//     if(order >= 2){
//       int err = with_xc(aux, [&](auto& xc){
//         return xc.eval_vxc_fxc_device(np, rho, sigma,
//                                       buf->vrho, buf->vsigma,
//                                       buf->v2rho2, buf->v2rhosigma, buf->v2sigma2,
//                                       qptr);
//       });
//       if(err) return err;
//     }
//     // Energy last for this component
//     if(buf->zk){
//       int err = with_xc(aux, [&](auto& xc){
//         return xc.eval_exc_device(np, rho, sigma, buf->zk, qptr);
//       });
//       if(err) return err;
//     }

//     // out += coef * buf  (per-field AXPY with correct dimensions)
//     axpy_gga_out(q, dim, out, buf, coef, order, static_cast<std::size_t>(np));
//   }

//   //q.wait(); // ensure accumulations are done
//   return 0;
// }

extern "C" int GDFT_xc_gga(
  void* stream_v,
  const xc_func_type *func, int np, const double *rho, const double *sigma,
  xc_gga_out_params *out, xc_gga_out_params* /*buf*/
){
  if(!func || !rho || !sigma || !out || np <= 0) return bad_args();

  const int order = detect_order(out);
  if(order < 0) return 0;
  if(order > 2){
    std::fprintf(stderr, "ExchCXX device: GGA order %d not implemented\n", order);
    return 2;
  }

  auto* stream = reinterpret_cast<sycl::queue*>(stream_v);
  double* eps    = out->zk;
  double* vrho   = out->vrho;
  double* vsigma = out->vsigma;
  double* v2rho2 = out->v2rho2;
  double* v2rs   = out->v2rhosigma;
  double* v2s2   = out->v2sigma2;

  //zero_gga_out(*stream, func, out, np, order);

  if(order >= 1){
    int err = with_xc(func, [&](auto& xc){
      xc.eval_exc_vxc_device(np, rho, sigma, eps, vrho, vsigma, stream);
    });
    if(err) return err;
  }
  if(order >= 2){
    int err = with_xc(func, [&](auto& xc){
      xc.eval_vxc_fxc_device(np, rho, sigma, vrho, vsigma, v2rho2, v2rs, v2s2, stream);
    });
    if(err) return err;
  }

  if(eps){
    int err = with_xc(func, [&](auto& xc){
      xc.eval_exc_device(np, rho, sigma, eps, stream);
    });
    if(err) return err;
  }
  return 0;
}

extern "C" int GDFT_xc_mgga(
  void* stream_v,
  const xc_func_type *func, int np,
  const double *rho, const double *sigma, const double *lapl, const double *tau,
  xc_mgga_out_params *out, xc_mgga_out_params* /*buf*/
){
  if(!func || !rho || !sigma || !tau || !out || np <= 0) return bad_args();

  const int order = detect_order(out);
  if(order < 0) return 0;
  if(order > 2){
    std::fprintf(stderr, "ExchCXX device: mGGA order %d not implemented\n", order);
    return 2;
  }

  auto* impl = get_impl(func);
  if(!impl) return 1;

  // If this functional doesn't need the Laplacian, pass nullptr for lapl and skip vlapl/its Hessians
  const bool need_lapl = impl->needs_lapl;
  const double* lapl_in = need_lapl ? lapl : nullptr;

  auto* stream = reinterpret_cast<sycl::queue*>(stream_v);

  // 1st-order outputs
  double* eps    = out->zk;
  double* vrho   = out->vrho;
  double* vsigma = out->vsigma;
  double* vlapl  = need_lapl ? out->vlapl : nullptr;
  double* vtau   = out->vtau;

  // 2nd-order outputs
  double* v2rho2      = out->v2rho2;
  double* v2rhosigma  = out->v2rhosigma;
  double* v2rholapl   = need_lapl ? out->v2rholapl  : nullptr;
  double* v2rhotau    = out->v2rhotau;
  double* v2sigma2    = out->v2sigma2;
  double* v2sigmalapl = need_lapl ? out->v2sigmalapl : nullptr;
  double* v2sigmatau  = out->v2sigmatau;
  double* v2lapl2     = need_lapl ? out->v2lapl2     : nullptr;
  double* v2lapltau   = (need_lapl ? out->v2lapltau  : nullptr);
  double* v2tau2      = out->v2tau2;

  if(order >= 1){
    int err = with_xc(func, [&](auto& xc){
      xc.eval_exc_vxc_device(np, rho, sigma, lapl_in, tau, eps, vrho, vsigma, vlapl, vtau, stream);
    });
    if(err) return err;
  }
  if(order >= 2){
    int err = with_xc(func, [&](auto& xc){
      xc.eval_vxc_fxc_device(np, rho, sigma, lapl_in, tau,
                             vrho, vsigma, vlapl, vtau,
                             v2rho2, v2rhosigma, v2rholapl, v2rhotau,
                             v2sigma2, v2sigmalapl, v2sigmatau,
                             v2lapl2, v2lapltau, v2tau2, stream);
    });
    if(err) return err;
  }

  if(eps){
    int err = with_xc(func, [&](auto& xc){
      xc.eval_exc_device(np, rho, sigma, lapl_in, tau, eps, stream);
    });
    if(err) return err;
  }

  return 0;
}
