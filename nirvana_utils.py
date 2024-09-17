# Nirvana dependencies
from distutils.dir_util import copy_tree
from shutil import rmtree
import os

try:
    import nirvana_dl
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
        copy_tree(snapshot_path, out, 
                #   preserve_symlinks=1, update=1,
                  )
        os.system(f"rm -rf {snapshot_path}/*")
        # os.system(f"tar -xf {out}/state -C {out}/")
    

def copy_out_to_snapshot(out, dump=True, clear_after_train=False):
    """ This function copies all files in the local "out" directory to "snapshot path".
        dump: If True, put these files into tar-archive "state" and 
              send it to the Python DL output.  
    """
    if nirvana_dl or True:
        snapshot_path = nirvana_dl.snapshot.get_snapshot_path() if nirvana_dl else "snapshot/"
        print(f"Copy {out} to the snapshot path: {snapshot_path}")

        # # Delete previous state to avoid memory explosion
        # print("====== Snapshot contents BEFORE dumping: =======")
        # os.system(f"ls -lahR {snapshot_path}")
        
        # print("====== Original filesystem out: =======")
        # os.system(f"ls -lahR {out}")

        # os.system(f"rm -rf {snapshot_path}/*")

        if os.path.exists(f"{snapshot_path}/state"):
            os.system(f"rm {snapshot_path}/state")


        copy_tree(out, snapshot_path, 
                #   preserve_symlinks=1, update=1,
                  )
        if clear_after_train:
            rmtree(os.path.join(snapshot_path, "checkpoints", "dataset"))
            os.remove(os.path.join(snapshot_path, "checkpoints", "load_metadata.json"))
            os.remove(os.path.join(snapshot_path, "checkpoints", "data_dict.pkl"))
            
        # print("====== Snapshot contents AFTER dumping: =======")
        
        # os.system(f"ls -lahR {snapshot_path}")
        if dump:
            # Make it visible in the Python DL output
            if nirvana_dl:
                nirvana_dl.snapshot.dump_snapshot(snapshot_path)
