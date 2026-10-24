"""Original sorting cell, PyBullet dynamics, RGB-D perception and suction control."""
from pathlib import Path
import json, math
import numpy as np
import pybullet as p
import pybullet_data
from perception import COLORS, LABELS, feature, ColorAI, detect

DT = 1/240
BIN_CENTERS = {'RED': [0.16,-0.56], 'BLUE': [0.49,-0.56],
               'YELLOW': [0.16,0.56], 'CHECK': [0.49,0.56]}
PALETTE = {'RED':[.88,.16,.24,1], 'BLUE':[.05,.66,.86,1],
           'YELLOW':[.95,.70,.06,1], 'CHECK':[.46,.35,.67,1]}

def box(half, xyz, color, mass=0, yaw=0):
    shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=half)
    visual=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=color)
    uid=p.createMultiBody(mass,shape,visual,xyz,p.getQuaternionFromEuler([0,0,yaw]))
    p.changeDynamics(uid,-1,lateralFriction=.8,restitution=.03)
    return uid

def connect():
    p.connect(p.DIRECT); p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81); p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSolverIterations=100, deterministicOverlappingPairs=1)

def train_ai(folder):
    """Render independent train/test RGB samples, then learn and measure."""
    folder=Path(folder); (folder/'models').mkdir(exist_ok=True)
    connect()
    box([1,1,.02],[0,0,-.02],[.58,.61,.63,1])
    obj=box([.04,.04,.03],[0,0,.03],[1,0,0,1])
    view=p.computeViewMatrix([0,0,1],[0,0,0],[0,1,0])
    proj=p.computeProjectionMatrixFOV(35,1,.05,3)
    def sample(seed, count, unknown=False):
        rng=np.random.default_rng(seed); xs=[]; ys=[]
        for k in range(4 if unknown else 3):
            if unknown and k != 3: continue
            for i in range(count):
                base = np.array([.52,.25,.64]) if unknown else np.array(COLORS[k])
                rgb=np.clip(base*rng.uniform(.75,1.08)+rng.normal(0,.018,3),.02,.98)
                p.changeVisualShape(obj,-1,rgbaColor=[*rgb,1])
                p.resetBasePositionAndOrientation(obj,[rng.uniform(-.1,.1),rng.uniform(-.1,.1),.03],p.getQuaternionFromEuler([0,0,rng.uniform(-3,3)]))
                im=p.getCameraImage(96,96,view,proj,renderer=p.ER_TINY_RENDERER,
                   lightDirection=[rng.uniform(-1,1),rng.uniform(-1,1),2],
                   lightAmbientCoeff=rng.uniform(.5,.8),lightDiffuseCoeff=rng.uniform(.2,.5),shadow=0)
                img=np.array(im[2],dtype=np.uint8).reshape(96,96,4)[...,:3]
                seg=np.array(im[4]).reshape(96,96)
                xs.append(feature(img[seg==obj])); ys.append(k)
        return np.array(xs),np.array(ys)
    x,y=sample(42,120); tx,ty=sample(2026,40); ux,uy=sample(773,30,True)
    ai=ColorAI().fit(x,y); ai.save(folder/'models/color_ai.npz')
    preds=[ai.predict(f) for f in tx]; un=[ai.predict(f) for f in ux]
    cm=np.zeros((3,4),int)
    for truth,pr in zip(ty,preds): cm[truth, ['RED','BLUE','YELLOW','CHECK'].index(pr['class'])]+=1
    metrics={'training_seed':42,'test_seed':2026,'train_samples':len(y),
      'test_samples':len(ty),'unknown_test_samples':len(uy),
      'test_accuracy':float(np.mean([v['class']==LABELS[t] for v,t in zip(preds,ty)])),
      'unknown_reject_rate':float(np.mean([v['class']=='CHECK' for v in un])),
      'confusion_columns':['RED','BLUE','YELLOW','CHECK'],'confusion_matrix':cm.tolist(),
      'radius':ai.radius,'initial_loss':ai.loss[0],'final_logged_loss':ai.loss[-1]}
    (folder/'evidence/training_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    np.savez_compressed(folder/'evidence/training_samples.npz',train_x=x,train_y=y,test_x=tx,test_y=ty,unknown_x=ux)
    p.disconnect(); return ai,metrics

class Cell:
    def __init__(self, ai, seed=7, capture=None):
        connect(); self.ai=ai; self.seed=seed; self.capture=capture
        self.time=0.; self.steps=0; self.state='READY'; self.current=None
        self.events=[]; self.counts={k:0 for k in BIN_CENTERS}; self.trials=[]
        self.attach=None; self.trail=[]; self.latest_camera=None
        box([1.5,1.5,.05],[0,0,-.11],[.13,.17,.21,1])
        box([.7,.78,.03],[.34,0,-.03],[.72,.76,.77,1])
        box([.245,.35,.002],[.59,0,.002],[.46,.51,.54,1])
        self.robot=p.loadURDF('kuka_iiwa/model.urdf',useFixedBase=True)
        self.orn=p.getQuaternionFromEuler([0,math.pi,0]); self.tool_len=.065
        self.home=np.array([.28,0,.62]); self.target=self.home.copy()
        for _ in range(12):
            q=p.calculateInverseKinematics(self.robot,6,self.home,self.orn,maxNumIterations=500,residualThreshold=1e-6)
            for j in range(7): p.resetJointState(self.robot,j,q[j])
        for label,(x,y) in BIN_CENTERS.items():
            color=PALETTE[label]
            box([.13,.115,.006],[x,y,.006],color)
            for dx,dy,hx,hy in [(0,-.12,.14,.005),(0,.12,.14,.005),(-.135,0,.005,.12),(.135,0,.005,.12)]:
                box([hx,hy,.045],[x+dx,y+dy,.045],[*np.array(color[:3])*.65,1])
        visual=p.createVisualShape(p.GEOM_CYLINDER,radius=.022,length=self.tool_len,rgbaColor=[.16,.2,.24,1])
        self.tool=p.createMultiBody(baseMass=0,baseVisualShapeIndex=visual)
        self.objects=[]; rng=np.random.default_rng(seed)
        positions=[[.44,-.25],[.68,-.23],[.46,-.03],[.69,.01],[.45,.23],[.69,.24]]
        labels=['RED','BLUE','YELLOW','BLUE','RED','CHECK']
        for i,(xy,label) in enumerate(zip(positions,labels)):
            rgb=np.clip(np.array(PALETTE[label][:3])*rng.uniform(.88,1.03),.02,.98)
            xyz=[xy[0]+rng.uniform(-.015,.015),xy[1]+rng.uniform(-.015,.015),.028]
            uid=box([.026,.026,.027],xyz,[*rgb,1],.055,rng.uniform(-.5,.5))
            self.objects.append({'id':uid,'label':label,'name':f'P{i+1:02}', 'done':False})
        self.sense_view=p.computeViewMatrix([.59,0,1.65],[.59,0,0],[0,1,0])
        self.sense_proj=p.computeProjectionMatrixFOV(48,1,.05,3)
        self.main_view=p.computeViewMatrix([1.62,-1.7,1.44],[.30,0,.23],[0,0,1])
        self.main_proj=p.computeProjectionMatrixFOV(45,16/10,.05,5)
        self.wait(.7,record=False)

    def tip(self):
        pos=np.array(p.getLinkState(self.robot,6,computeForwardKinematics=1)[4])
        return pos-np.array([0,0,self.tool_len])

    def step(self, record=True):
        q=p.calculateInverseKinematics(self.robot,6,self.target,self.orn,maxNumIterations=80,residualThreshold=1e-5)
        p.setJointMotorControlArray(self.robot,list(range(7)),p.POSITION_CONTROL,
           targetPositions=q[:7],forces=[500]*7,positionGains=[.25]*7,velocityGains=[1]*7)
        p.stepSimulation(); self.time+=DT; self.steps+=1
        flange=p.getLinkState(self.robot,6,computeForwardKinematics=1)
        pose=p.multiplyTransforms(flange[4],flange[5],[0,0,self.tool_len/2],[0,0,0,1])
        # Cylinder is visual only; grasp loads are carried by the real fixed constraint.
        p.resetBasePositionAndOrientation(self.tool,*pose)
        if self.steps%8==0:
            self.trail.append(self.tip().tolist())
        if self.steps%16==0 and record and self.capture:
            self.capture(self)

    def wait(self, seconds, record=True):
        for _ in range(round(seconds/DT)): self.step(record)

    def move(self, xyz, seconds):
        start=self.target.copy(); dest=np.array(xyz,float)
        n=round(seconds/DT)
        for k in range(n):
            t=(k+1)/n; s=10*t**3-15*t**4+6*t**5
            self.target=start+(dest-start)*s; self.step()
        self.wait(.15)

    def sense(self):
        im=p.getCameraImage(320,320,self.sense_view,self.sense_proj,renderer=p.ER_TINY_RENDERER,shadow=0,
                            lightDirection=[1,-1,3],lightAmbientCoeff=.7,lightDiffuseCoeff=.3)
        rgb=np.array(im[2],np.uint8).reshape(320,320,4)[...,:3]
        depth=np.array(im[3]).reshape(320,320)
        detections=detect(rgb,depth,self.sense_view,self.sense_proj,self.ai)
        self.latest_camera=rgb
        return detections

    def change(self, state, detail=''):
        self.state=state; self.events.append({'t':round(self.time,3),'state':state,'detail':detail})

    def run(self):
        self.change('SCAN'); self.detections=self.sense(); self.wait(3)
        if len(self.detections)!=6:
            raise RuntimeError(f'Expected six RGB blobs, detected {len(self.detections)}')
        for d in self.detections:
            self.current=d; xyz=np.array(d['xyz']); label=d['class']
            self.change('CLASSIFY',label); self.wait(1.3)
            # Pose comes only from RGB-D. Ground truth IDs below are used after
            # physical proximity checks for suction actuation and evaluation.
            self.change('APPROACH'); self.move([xyz[0],xyz[1],.30],1.5)
            self.change('DESCEND'); self.move([xyz[0],xyz[1],xyz[2]+self.tool_len+.006],1.2)
            tip=self.tip()
            near=[]
            for obj in self.objects:
                if obj['done']: continue
                pos=np.array(p.getBasePositionAndOrientation(obj['id'])[0])
                distance=np.linalg.norm(tip-(pos+[0,0,.027]))
                if distance < .027: near.append((distance,obj))
            if not near: raise RuntimeError(f'No object within suction range: tip={tip}, observation={xyz}')
            distance,obj=min(near,key=lambda v:v[0]); uid=obj['id']
            flange=p.getLinkState(self.robot,6,computeForwardKinematics=1)
            inv=p.invertTransform(flange[4],flange[5]); pos,orn=p.getBasePositionAndOrientation(uid)
            rel=p.multiplyTransforms(*inv,pos,orn)
            self.attach=p.createConstraint(self.robot,6,uid,-1,p.JOINT_FIXED,[0,0,0],rel[0],[0,0,0],parentFrameOrientation=rel[1])
            p.changeConstraint(self.attach,maxForce=80)
            self.change('SUCTION'); self.wait(.5)
            self.change('LIFT'); self.move([xyz[0],xyz[1],.36],1.25)
            lifted=float(p.getBasePositionAndOrientation(uid)[0][2])>.20
            self.change('TRANSFER'); bx,by=BIN_CENTERS[label]
            n=self.counts[label]; slot_x=bx+(-.052 if n%2==0 else .052)
            self.move([slot_x,by,.36],1.9)
            self.change('PLACE'); self.move([slot_x,by,.22],.9)
            p.removeConstraint(self.attach); self.attach=None
            self.change('RELEASE'); self.wait(.9)
            pos=np.array(p.getBasePositionAndOrientation(uid)[0])
            placed=bool(abs(pos[0]-bx)<.12 and abs(pos[1]-by)<.10 and .02<pos[2]<.13)
            correct=label==obj['label']
            self.trials.append({'object':obj['name'],'true_class':obj['label'],'predicted_class':label,
               'score':d['score'],'distance':d['distance'],'suction_gap_m':float(distance),
               'lifted':lifted,'in_target_bin':placed,'correct_class':correct,
               'final_xyz':pos.tolist(),'success':bool(lifted and placed and correct)})
            if placed: self.counts[label]+=1
            obj['done']=True
            self.change('VERIFY','PASS' if placed and correct and lifted else 'FAIL'); self.wait(.7)
            self.move([slot_x,by,.39],.8)
        self.change('RETURN'); self.move(self.home,1.5)
        self.change('COMPLETE'); self.wait(4)
        return {'seed':self.seed,'physics_dt':DT,'simulation_seconds':round(self.time,3),
                'objects':len(self.trials),'successful':sum(t['success'] for t in self.trials),
                'bins':self.counts,'trials':self.trials,'events':self.events,
                'trajectory':self.trail}

    def render(self,width=1200,height=750):
        im=p.getCameraImage(width,height,self.main_view,self.main_proj,renderer=p.ER_TINY_RENDERER,
             shadow=1,lightDirection=[-2,-3,5],lightAmbientCoeff=.6,lightDiffuseCoeff=.45)
        return np.array(im[2],np.uint8).reshape(height,width,4)[...,:3]

    def close(self): p.disconnect()
