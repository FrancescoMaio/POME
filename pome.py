import sys
import subprocess
import os

def print_help():
    print("""
Usage: python pome.py <target_name> <mode>

<target_name> : Name of the target to be reduced.
<mode>        : One of the following observation modes:
                - vortex     : Coronagraphic mode using the vortex
                - coro_app       : Coronagraphic mode using vAPP 
                - img_pupil  : Imaging in pupil-stabilized mode 
                - img_field  : Imaging in field-stabilized mode (not yet supported)

Example:
    python3 pome.py HD100546 vortex
    """)
    sys.exit(0)

def print_banner():
    print(r"""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║                 ██████╗  ██████╗ ███╗   ███╗███████╗               ║
║                 ██╔══██╗██╔═══██╗████╗ ████║██╔════╝               ║
║                 ██████╔╝██║   ██║██╔████╔██║█████╗                 ║
║                 ██╔═══╝ ██║   ██║██║╚██╔╝██║██╔══╝                 ║
║                 ██║     ╚██████╔╝██║ ╚═╝ ██║███████╗               ║
║                 ╚═╝      ╚═════╝ ╚═╝     ╚═╝╚══════╝               ║
║                                                                    ║
║         Pipeline for Observation in Mid-infrared with ERIS         ║
╚════════════════════════════════════════════════════════════════════╝
""")

PATH2DATA = os.path.dirname(os.path.abspath(__file__))
PATH2DATA = PATH2DATA+'/'

if __name__ == "__main__":
    print_banner()
    if len(sys.argv) != 3 or sys.argv[1] in ["-h", "--help"]:
        print_help()

    TARGETNAME = sys.argv[1]
    MODE = sys.argv[2].lower()

    os.makedirs(os.path.join(PATH2DATA, 'reduced', TARGETNAME, 'image'), exist_ok=True)
    print(f"Directory 'image' created (or already exists).")

    if MODE == "vortex":
        scripts = [
            "eris_calib_dark.py",
            "eris_calib_dark_flat.py",
            "eris_calib_flat.py",
            "eris_calib_dark_psf.py",
            "eris_elab_psf.py",
            "eris_elab_sky.py",
            "eris_elab_obj.py"
        ]

        # Check if all the required scripts exist
        for script in scripts:
            script_path = os.path.join(PATH2DATA, script)
            if not os.path.exists(script_path):
                print(f"Error: The script '{script}' is missing from the directory.")
                sys.exit(1)

        for script in scripts:
            print(f"Running {script} with arguments: {PATH2DATA} {TARGETNAME}")
            subprocess.run(["python3", script, PATH2DATA, TARGETNAME], check=True)
            print(f"Finished {script}\n")

    elif MODE == "coro_app":
        scripts = [
            "eris_calib_dark.py",
            "eris_calib_dark_flat.py",
            "eris_calib_flat.py",
            "vapp_eris_background.py",
            "vapp_eris_elab.py",
            "vapp_eris_dh_estraction.py"
        ]
    
        # Check if all the required scripts exist
        for script in scripts:
            script_path = os.path.join(PATH2DATA, script)
            if not os.path.exists(script_path):
                print(f"Error: The script '{script}' is missing from the directory.")
                sys.exit(1)
    
        for script in scripts:
            print(f"Running {script} with arguments: {PATH2DATA} {TARGETNAME}")
            subprocess.run(["python3", script, PATH2DATA, TARGETNAME], check=True)
            print(f"Finished {script}\n")
            
            
    elif MODE == "img_pupil":
        scripts = [
            "eris_calib_dark.py",
            "eris_calib_dark_flat.py",
            "eris_calib_flat.py",
            "eris_background_img_pupil.py",
            "eris_elab_img_pupil.py"
        ]
    
        # Check if all the required scripts exist
        for script in scripts:
            script_path = os.path.join(PATH2DATA, script)
            if not os.path.exists(script_path):
                print(f"Error: The script '{script}' is missing from the directory.")
                sys.exit(1)
    
        for script in scripts:
            print(f"Running {script} with arguments: {PATH2DATA} {TARGETNAME}")
            subprocess.run(["python3", script, PATH2DATA, TARGETNAME], check=True)
            print(f"Finished {script}\n")
            
    elif MODE == "img_field":
        scripts = [
            "eris_calib_dark.py",
            "eris_calib_dark_flat.py",
            "eris_calib_flat.py"
        ]
    
        # Check if all the required scripts exist
        for script in scripts:
            script_path = os.path.join(PATH2DATA, script)
            if not os.path.exists(script_path):
                print(f"Error: The script '{script}' is missing from the directory.")
                sys.exit(1)
    
        for script in scripts:
            print(f"Running {script} with arguments: {PATH2DATA} {TARGETNAME}")
            subprocess.run(["python3", script, PATH2DATA, TARGETNAME], check=True)
            print(f"Finished {script}\n")

    else:
        print(f"Unknown mode: {MODE}\n")
        print_help()
