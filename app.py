"""
VigilanceAI — Interface Hugging Face Spaces (Gradio)
"""
import cv2, numpy as np, tempfile, os, time, gradio as gr
from collections import deque
from ultralytics import YOLO

PERSON_CLASS=0
VEHICLE_CLASSES={2:"Car",3:"Motorcycle",5:"Bus",7:"Truck"}
OBJECT_CLASSES={24:"Backpack",26:"Handbag",28:"Suitcase",39:"Bottle",43:"Knife",76:"Scissors"}
SUSPICIOUS_OBJECTS={43,76}
CONF=0.40; TRAIL_LEN=40; MAX_HIST=20; ECHELLE=0.05
SCORE_GENTIL=35; SCORE_SUSPECT=65
TRAIL_PALETTE=[(0,255,100),(0,170,255),(255,200,0),(255,80,0),(200,0,255),(0,255,255)]
C_GREEN=(0,255,100); C_ORANGE=(0,140,255); C_RED=(0,0,255); C_WHITE=(240,240,240)

model=YOLO("yolov8n.pt")

class Tracker:
    def __init__(self): self.next_id=0; self.tracks={}
    @staticmethod
    def iou(b1,b2):
        xa=max(b1[0],b2[0]);ya=max(b1[1],b2[1])
        xb=min(b1[2],b2[2]);yb=min(b1[3],b2[3])
        inter=max(0,xb-xa)*max(0,yb-ya)
        a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
        return inter/(a1+a2-inter+1e-6)
    def update(self,dets,t):
        for tid in list(self.tracks): self.tracks[tid]['miss']+=1
        matched=set(); results=[]
        for det in dets:
            x1,y1,x2,y2=det; box=(x1,y1,x2,y2)
            bi,bd=0,None
            for tid,tr in self.tracks.items():
                if tid in matched: continue
                sc=self.iou(box,tr['box'])
                if sc>bi: bi,bd=sc,tid
            if bi>=0.25 and bd is not None: tid=bd
            else:
                tid=self.next_id; self.next_id+=1
                self.tracks[tid]={'box':box,'hist':[],'miss':0,'trail':deque(maxlen=TRAIL_LEN)}
            tr=self.tracks[tid]; cx=(x1+x2)/2; cy=(y1+y2)/2
            tr['hist'].append((cx,cy,t))
            if len(tr['hist'])>MAX_HIST: tr['hist'].pop(0)
            tr['box']=box; tr['miss']=0; tr['trail'].append((int(cx),int(cy)))
            matched.add(tid); results.append((tid,box,cx,cy,tr['hist'],tr['trail']))
        for tid in list(self.tracks):
            if self.tracks[tid]['miss']>20: del self.tracks[tid]
        return results

def analyze_posture(history):
    if len(history)<3: return "inconnu",0,False
    positions=[(h[0],h[1]) for h in history]; times=[h[2] for h in history]
    speeds=[]
    for i in range(1,len(positions)):
        dt=times[i]-times[i-1]
        if dt<1e-4: continue
        dx=positions[i][0]-positions[i-1][0]; dy=positions[i][1]-positions[i-1][1]
        speeds.append(np.sqrt(dx**2+dy**2)/dt)
    avg=np.mean(speeds) if speeds else 0
    fuite=avg>80
    score=0; posture="normal"
    if avg>120: score=50; posture="FUITE"
    elif avg>60: score=30; posture="course"
    elif avg<5 and len(history)>15: score=20; posture="surveillance"
    else: posture="marche"
    return posture,min(score,80),fuite

def threat_score(posture_score,has_obj):
    s=posture_score*0.6+(30 if has_obj else 0)
    return min(round(s),100)

def get_statut(score):
    if score>=SCORE_SUSPECT: return "SUSPECT",C_RED
    elif score>=SCORE_GENTIL: return "NEUTRE",C_ORANGE
    return "GENTIL",C_GREEN

def draw_person(frame,x1,y1,x2,y2,pid,score,statut,color,posture,objet,in_zone):
    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
    bgr=color
    cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)
    cs,lw=10,3
    for px,py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
        dx=cs if px==x1 else -cs; dy=cs if py==y1 else -cs
        cv2.line(frame,(px,py),(px+dx,py),bgr,lw); cv2.line(frame,(px,py),(px,py+dy),bgr,lw)
    flags=(" [ZONE]" if in_zone else "")+(" ["+objet+"]" if objet else "")
    txt=f"#{pid} {statut} {score}%{flags}"
    (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.46,1)
    lx,ly=max(x1,0),max(y1-20,0)
    cv2.rectangle(frame,(lx,ly),(lx+tw+6,ly+th+4),bgr,-1)
    cv2.putText(frame,txt,(lx+3,ly+th+1),cv2.FONT_HERSHEY_SIMPLEX,0.46,(0,0,0),1,cv2.LINE_AA)
    bw=x2-x1; ratio=score/100
    bc=C_RED if ratio>.65 else C_ORANGE if ratio>.35 else C_GREEN
    cv2.rectangle(frame,(x1,y2+2),(x2,y2+4),(40,40,40),-1)
    cv2.rectangle(frame,(x1,y2+2),(x1+int(bw*ratio),y2+4),bc,-1)
    if posture:
        cv2.putText(frame,posture,(x1,y2+16),cv2.FONT_HERSHEY_SIMPLEX,0.33,(180,180,180),1)

def draw_trails(frame,tracker):
    for tid,tr in tracker.tracks.items():
        pts=list(tr['trail'])
        if len(pts)<2: continue
        color=TRAIL_PALETTE[tid%len(TRAIL_PALETTE)]
        for i in range(1,len(pts)):
            alpha=i/len(pts); c=tuple(int(v*alpha) for v in color)
            cv2.line(frame,pts[i-1],pts[i],c,max(1,int(alpha*3)),cv2.LINE_AA)

def draw_hud(frame,fps,total,suspects,neutres,gentils,crowd,fn,viol):
    H,W=frame.shape[:2]
    ov=frame.copy()
    cv2.rectangle(ov,(6,6),(225,185),(10,12,16),-1)
    cv2.addWeighted(ov,0.78,frame,0.22,0,frame)
    cv2.rectangle(frame,(6,6),(225,185),(0,0,200),1)
    lines=[
        ("VIGILANCE AI",       (0,100,255),0.48,2),
        (f"FRAME  : {fn}",     C_GREEN,    0.36,1),
        (f"FPS    : {fps:.1f}",C_GREEN,    0.36,1),
        (f"TOTAL  : {total}",  C_WHITE,    0.36,1),
        (f"GENTILS: {gentils}",C_GREEN,    0.36,1),
        (f"NEUTRES: {neutres}",(0,200,255),0.36,1),
        (f"SUSPECTS:{suspects}",C_RED,     0.36,1),
        (f"INFRAC : {viol}",   C_RED if viol>0 else C_GREEN,0.36,1),
        (f"FOULE  : {'ALERTE!' if crowd else 'OK'}",
         C_RED if crowd else C_GREEN,      0.36,1),
    ]
    for i,(txt,col,sc,th) in enumerate(lines):
        cv2.putText(frame,txt,(13,26+i*17),cv2.FONT_HERSHEY_SIMPLEX,sc,col,th,cv2.LINE_AA)
    cv2.circle(frame,(W-18,15),6,C_RED,-1)
    cv2.putText(frame,"LIVE",(W-58,20),cv2.FONT_HERSHEY_SIMPLEX,0.42,C_RED,1)
    if crowd:
        cv2.rectangle(frame,(0,0),(W,36),(0,0,140),-1)
        cv2.putText(frame,"ALERTE FOULE — SURPOPULATION",
                    (8,24),cv2.FONT_HERSHEY_SIMPLEX,0.60,C_WHITE,2)

def process_video(video_path, seuil_suspect, show_trails, max_frames, progress=gr.Progress()):
    if video_path is None: return None,"❌ Aucune vidéo."
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None,"❌ Impossible d'ouvrir."
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src=cap.get(cv2.CAP_PROP_FPS) or 30
    total_f=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit=min(int(max_frames),total_f) if total_f>0 else int(max_frames)
    out=tempfile.mktemp(suffix="_vigilance.mp4")
    writer=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*'mp4v'),fps_src,(W,H))
    tracker=Tracker()
    frame_num=0; violations=0
    total_suspects=0; total_gentils=0; total_neutres=0
    all_ids=set()
    while frame_num<limit:
        ret,frame=cap.read()
        if not ret: break
        frame_num+=1; t_now=frame_num/fps_src
        if show_trails: draw_trails(frame,tracker)
        results=model(frame,classes=[PERSON_CLASS]+list(VEHICLE_CLASSES.keys())+list(OBJECT_CLASSES.keys()),
                      conf=CONF,iou=0.45,verbose=False)[0]
        pdets=[]; vdets=[]; odets=[]
        for box in results.boxes:
            x1,y1,x2,y2=box.xyxy[0].tolist(); cls=int(box.cls[0]); conf=float(box.conf[0])
            if cls==PERSON_CLASS: pdets.append((x1,y1,x2,y2))
            elif cls in VEHICLE_CLASSES: vdets.append((x1,y1,x2,y2,cls))
            elif cls in OBJECT_CLASSES: odets.append((x1,y1,x2,y2,cls))
        tracked=tracker.update(pdets,t_now)
        ns=0; nn=0; ng=0
        for pid,box,cx,cy,hist,trail in tracked:
            all_ids.add(pid)
            x1,y1,x2,y2=box
            posture,pscore,fuite=analyze_posture(hist)
            obj=None
            for ox1,oy1,ox2,oy2,ocls in odets:
                ocx=(ox1+ox2)/2; ocy=(oy1+oy2)/2
                if x1<=ocx<=x2 and y1<=ocy<=y2:
                    if ocls in SUSPICIOUS_OBJECTS: obj=OBJECT_CLASSES[ocls]
            score=threat_score(pscore,obj is not None)
            if score>=seuil_suspect: score=max(score,65)
            statut,color=get_statut(score)
            if statut=="SUSPECT": ns+=1; violations+=1
            elif statut=="NEUTRE": nn+=1
            else: ng+=1
            draw_person(frame,x1,y1,x2,y2,pid,score,statut,color,posture,obj,False)
            if fuite and statut=="SUSPECT":
                for vx1,vy1,vx2,vy2,vcls in vdets:
                    vcx=(vx1+vx2)/2; vcy=(vy1+vy2)/2
                    if np.sqrt((vcx-cx)**2+(vcy-cy)**2)<200:
                        cv2.rectangle(frame,(int(vx1),int(vy1)),(int(vx2),int(vy2)),(0,0,255),3)
                        cv2.putText(frame,f"VEHICULE FUITE #{pid}",(int(vx1),int(vy1)-8),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        for vx1,vy1,vx2,vy2,vcls in vdets:
            cv2.rectangle(frame,(int(vx1),int(vy1)),(int(vx2),int(vy2)),(255,200,0),1)
        total_suspects+=ns; total_neutres+=nn; total_gentils+=ng
        crowd=len(tracked)>=10
        draw_hud(frame,fps_src,len(tracked),ns,nn,ng,crowd,frame_num,violations)
        writer.write(frame)
        progress(frame_num/limit,desc=f"Frame {frame_num}/{limit}")
    cap.release(); writer.release()
    rapport=f"""
## 📊 Rapport VigilanceAI

| Métrique | Valeur |
|----------|--------|
| 🎬 Frames traitées | {frame_num} |
| 👥 Individus uniques | {len(all_ids)} |
| ✅ Gentils | {total_gentils} |
| ⚠️ Neutres | {total_neutres} |
| 🚨 Suspects | {total_suspects} |
| ⚠️ Violations totales | {violations} |
"""
    return out,rapport

with gr.Blocks(
    title="VigilanceAI — Surveillance Comportementale",
    theme=gr.themes.Base(primary_hue="red"),
    css="h1{text-align:center;font-family:monospace;letter-spacing:3px} body{background:#080a0d}"
) as demo:
    gr.Markdown("""
# 👁️ VIGILANCE AI
### Système de surveillance comportementale — YOLOv8
Détecte, analyse et classe chaque individu : Gentil / Neutre / Suspect
""")
    with gr.Row():
        with gr.Column(scale=1):
            video_input=gr.Video(label="📹 Vidéo à analyser",height=280)
            with gr.Accordion("⚙️ Paramètres",open=True):
                seuil=gr.Slider(40,90,value=65,step=5,label="🚨 Seuil score SUSPECT (%)")
                trails=gr.Checkbox(value=True,label="🎨 Trainées de trajectoire")
                max_frames=gr.Slider(50,500,value=200,step=50,label="🎬 Frames à traiter")
            btn=gr.Button("🚀 Analyser",variant="primary",size="lg")
        with gr.Column(scale=1):
            video_output=gr.Video(label="🎯 Résultat annoté",height=280)
            rapport_out=gr.Markdown()
    gr.Markdown("""
---
### 🎯 Ce que détecte VigilanceAI
- 👤 **Personnes** avec score de menace 0–100%
- 🏃 **Comportement** : marche / course / fuite / surveillance
- 🎒 **Objets suspects** : couteau, ciseaux, sacs
- 🚗 **Véhicule de fuite** si suspect en mouvement
- 👥 **Analyse de foule** + alerte surpopulation

### 🔴 Niveaux de menace
| Score | Statut | Couleur |
|-------|--------|---------|
| 0–34% | GENTIL | 🟢 Vert |
| 35–64% | NEUTRE | 🟠 Orange |
| 65–100% | SUSPECT | 🔴 Rouge |

*Propulsé par YOLOv8 · Fait par **Mayele***
""")
    btn.click(fn=process_video,
              inputs=[video_input,seuil,trails,max_frames],
              outputs=[video_output,rapport_out])

if __name__=="__main__":
    demo.launch()
