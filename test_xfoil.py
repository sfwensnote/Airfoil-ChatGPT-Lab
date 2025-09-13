import subprocess, os, tempfile

def test_xfoil(naca_code="2412", Re=1e6, Mach=0.0, Ncrit=9,
               alpha_start=0, alpha_end=10, alpha_step=1):
    exe = os.path.abspath("xfoil.exe")
    if not os.path.exists(exe):
        raise FileNotFoundError(f"❌ xfoil.exe not found at {exe}")

    with tempfile.TemporaryDirectory() as td:
        pol_path = os.path.join(td, "polar.out")

        script = f"""
NACA {naca_code}
PANE
OPER
VISC {Re:.3e}
MACH {Mach:.4f}
VPAR
N {int(Ncrit)}

PACC
{pol_path}

ASEQ {alpha_start:.1f} {alpha_end:.1f} {alpha_step:.1f}
PACC

QUIT
"""

        result = subprocess.run(
            [exe],
            input=script,
            text=True,
            capture_output=True,
            cwd=os.getcwd()  # ✅ 改成当前目录，而不是临时目录
        )

        print("=== STDOUT (first 400) ===")
        print(result.stdout[:400])
        print("=== STDERR (first 400) ===")
        print(result.stderr[:400])

        if os.path.exists(pol_path):
            print(f"✅ polar.out generated: {pol_path}")
            with open(pol_path) as f:
                print("=== polar.out (head) ===")
                for line in f.readlines()[:20]:
                    print(line.strip())
        else:
            print("❌ polar.out not generated")

if __name__ == "__main__":
    test_xfoil()
