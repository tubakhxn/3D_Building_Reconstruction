# Developer/Creator: tubakhxn
import subprocess,sys,os
os.environ.setdefault('F0','1050')
for cmd in (['step1_frames.py'],['step2_features.py'],['step3_sfm.py'],['step4_mvs.py'],['step4b_fuse.py'],['step5_mesh.py'],['render_video.py','all']):
    print('>>',*cmd,flush=True);subprocess.run([sys.executable,*cmd],check=True)
