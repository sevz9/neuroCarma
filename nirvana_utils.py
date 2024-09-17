# Nirvana dependencies
try:
    import nirvana_dl
    from distutils.dir_util import copy_tree
    from shutil import rmtree
    import os
except ImportError:
    nirvana_dl = None


def copy_snapshot_to_out(out):
    """ The preempted run transfers its "state" to the restarted run through "snapshot path".
        "state" is a tar-archive that contains all files put into "snapshot path" by the preempted run.
        This function moves the files in the "state" archive to you local "out" dir.
    """
    if nirvana_dl:
        snapshot_path = nirvana_dl.snapshot.get_snapshot_path()
        print(f"Copy the previous state from {snapshot_path} to {out}")
        copy_tree(snapshot_path, out)
        os.system(f"rm -rf {snapshot_path}/*")
        # os.system(f"tar -xf {out}/state -C {out}/")
    

def copy_out_to_snapshot(out, dump=True):
    """ This function copies all files in the local "out" directory to "snapshot path".
        dump: If True, put these files into tar-archive "state" and 
              send it to the Python DL output.  
    """
    if nirvana_dl:
        print("========================= DUMPING SNAPSHOT =========================")
        snapshot_path = nirvana_dl.snapshot.get_snapshot_path()
        print(f"Copy {out} to the snapshot path: {snapshot_path}")

        # Delete previous state to avoid memory explosion
        
        
        print("====== Original filesystem out: =======")
        os.system(f"ls -lahR {out}")

        # os.system(f"rm -rf {snapshot_path}/*")

        if os.path.exists(f"{snapshot_path}/state"):
            # os.system(f"rm {snapshot_path}/state")
            rmtree(snapshot_path)
            os.makedirs(snapshot_path, exist_ok=True)
        
        print("====== Snapshot contents: =======")
        os.system(f"ls -lahR {snapshot_path}")
            
        copy_tree(out, snapshot_path)

        if dump:
            # Make it visible in the Python DL output
            nirvana_dl.snapshot.dump_snapshot(snapshot_path)
        print("========================= SNAPSHOT DUMPED =========================")