import h5py

f_data = h5py.File("fcsn_tvsum.h5")

video_dict = {6912: '-esJrBWj2d8.mp4', 3532: '0tmA_C6XwfM.mp4', 5742: '37rzWOQsNIw.mp4', 4853: '3eYKfiOEJNs.mp4', 4005: '4wU_LUjG5Ic.mp4', 3312: '91IHQYk1IQM.mp4', 4688: '98MoyGZKHXc.mp4', 10597: 'AwmHb44_ouw.mp4', 13511: 'Bhxk-O1Y7Ho.mp4', 15307: 'E11zDS9XGzg.mp4', 2941: 'EE-bNr36nyA.mp4', 5939: 'EYqVtI9YWJA.mp4', 4356: 'GsAD1KT1xo8.mp4', 9671: 'HT5vyqe0Xaw.mp4', 5846: 'Hl-__g2gn_A.mp4', 14019: 'J0nA4VgnoCo.mp4', 3802: 'JKpqYvAdIsw.mp4', 4304: 'JgHubY5Vw3Y.mp4', 6241: 'LRw_obCPUt0.mp4', 4740: 'NyBmCxDoHJU.mp4', 6580: 'PJrm840pAUI.mp4', 10917: 'RBCABdttQmI.mp4', 4166: 'Se3oxnaPsz0.mp4', 5412: 'VuWGsYPqAX8.mp4', 9535: 'WG0MBPpPC6I.mp4', 7959: 'WxtbjNsCQ8A.mp4', 5631: 'XkqCExn6_Us.mp4', 3327: 'XzYM3PfTM4w.mp4', 9731: 'Yi4Ij2NM7U4.mp4', 4463: '_xMr-HKMfVA.mp4', 3995: 'akI8YFjEmUw.mp4', 5661: 'b626MiF1ew4.mp4', 3705: 'byxOvuiIJV0.mp4', 19406: 'cjibtmSLxQ4.mp4', 4931: 'eQu1rNs0an0.mp4', 17527: 'fWutDQy1nnY.mp4', 7210: 'gzDbaEs1Rlg.mp4', 4700: 'i3wAGJaaktw.mp4', 2500: 'iVt07TCkFM0.mp4', 5971: 'jcoYJXDG9sw.mp4', 3896: 'kLxoNp-UchI.mp4', 11414: 'oDXZc0tZe04.mp4', 8073: 'qqR6AEXwxoQ.mp4', 4468: 'sTEELN-vY30.mp4', 4009: 'uGu_10sucQo.mp4', 9870: 'vdmoEJ5YbrQ.mp4', 13365: 'xmEERLqJ2kU.mp4', 7010: 'xwqBXPGE9pQ.mp4', 4324: 'xxdtq8mxegs.mp4', 8281: 'z_6gVvQb2d0.mp4'}
dict = {}

for i in range(1, 51):
    video_path = "video_" + str(i)
    frames = f_data[video_path]['length'].value
    if frames == 9534 or frames == 4165:
        frames += 1
    dict[video_path] = video_dict[frames]

print(dict)