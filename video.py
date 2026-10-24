"""1080p, 30 fps, 1x-simulation-time documentary with Japanese captions."""
from pathlib import Path
import json,subprocess,shutil,os
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from simulation import BIN_CENTERS

INK='#122b3d'; MUTED='#5f7480'; BG='#edf2f2'; WHITE='#ffffff'
COL={'RED':'#dd344d','BLUE':'#0aabcf','YELLOW':'#e5b027','CHECK':'#8d70b5'}
JA={'RED':'赤 / 回収 A','BLUE':'青 / 回収 B','YELLOW':'黄 / 回収 C','CHECK':'待確認'}
STATES={
 'READY':('準備','RGB-Dカメラと7軸アームを使った、自動仕分けの実験です。'),
 'SCAN':('01  撮像・検出','RGB画像から物体を検出し、深度画像から把持位置を求めます。'),
 'CLASSIFY':('02  AIで分類','学習済み分類器で色を推定。学習分布から遠い物体は「待確認」へ。'),
 'APPROACH':('03  把持位置へ','画像から得た座標を使い、逆運動学でアームを動かします。'),
 'DESCEND':('04  下降','吸着点へゆっくり下降。物体との距離を確認してから接続します。'),
 'SUCTION':('05  吸着','吸着は距離判定付きの固定拘束で近似。真空圧の計算は行いません。'),
 'LIFT':('06  持ち上げ','重力と関節モーターを使った物理シミュレーションで持ち上げます。'),
 'TRANSFER':('07  搬送','AIが選んだ回収箱へ搬送。認識とアーム制御を分けて実装しました。'),
 'PLACE':('08  投入位置へ','箱の縁を避けて下降し、同じ箱の2個目は投下位置をずらします。'),
 'RELEASE':('09  吸着解除','拘束を外すと、物体は重力で落下して箱の底に着地します。'),
 'VERIFY':('10  結果確認','持ち上げ・箱内への着地・分類の一致を実行ログに記録します。'),
 'RETURN':('ホームへ復帰','6個の搬送が完了。アームをホーム位置へ戻します。'),
 'COMPLETE':('タスク完了','赤2・青2・黄1を仕分け、未学習の紫1を待確認箱へ搬送しました。')}

def fonts():
    candidates=[Path(os.environ.get('ECOSORT_FONT','')),Path('C:/Windows/Fonts/YuGothM.ttc'),
                Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    path=next((p for p in candidates if p.is_file()),None)
    if path is None: raise RuntimeError('Set ECOSORT_FONT to a Japanese TTF/TTC font.')
    return {s:ImageFont.truetype(str(path),s) for s in [20,22,24,26,28,30,32,36,40,48,52,64,76]}

class Movie:
    def __init__(self,path,root):
        self.path=Path(path); self.root=Path(root); self.f=fonts(); self.n=0
        self.metrics=json.loads((self.root/'evidence/training_metrics.json').read_text())
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg: raise RuntimeError('FFmpeg is required on PATH.')
        self.path.parent.mkdir(exist_ok=True,parents=True)
        self.err=open(self.root/'evidence/ffmpeg.log','w',encoding='utf-8')
        self.proc=subprocess.Popen([ffmpeg,'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080',
              '-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
              '-movflags','+faststart',str(self.path)],stdin=subprocess.PIPE,stderr=self.err)
        self.last_state=None

    def text(self,d,xy,t,size=28,color=INK): d.text(xy,t,font=self.f[size],fill=color)
    def rounded(self,d,rect,fill=WHITE,r=18): d.rounded_rectangle(rect,radius=r,fill=fill)

    def image(self,cell):
        frame=Image.new('RGB',(1920,1080),BG); d=ImageDraw.Draw(frame)
        self.text(d,(48,30),'EcoSort Mini',64)
        self.text(d,(52,112),'AI視覚と7軸アームによる、色分け回収物の自動仕分け',28,MUTED)
        self.text(d,(1430,45),'PAI FINAL  /  Sabercon',28)
        self.text(d,(1430,91),'PyBullet simulation  •  通常速度 1x',22,MUTED)
        scene=Image.fromarray(cell.render(832,520)).resize((1248,780),Image.Resampling.LANCZOS)
        frame.paste(scene,(48,186))
        d=ImageDraw.Draw(frame)
        d.rounded_rectangle([48,186,1296,966],radius=12,outline='#c6d1d4',width=2)
        # Bin labels are projected from world coordinates, not drawn onto physics.
        vp=np.array(cell.main_proj).reshape(4,4,order='F') @ np.array(cell.main_view).reshape(4,4,order='F')
        for label,(x,y) in BIN_CENTERS.items():
            q=vp @ [x,y,.055,1]; q=q[:3]/q[3]
            px=48+(q[0]+1)*624; py=186+(1-q[1])*390
            self.rounded(d,[px-55,py+28,px+75,py+63],INK,5)
            self.text(d,(px-45,py+29),label if label!='CHECK' else 'CHECK',20,WHITE)
        self.rounded(d,[72,206,360,252],WHITE,8)
        self.text(d,(86,210),f'SIM  {cell.time:05.1f} s   |   240 Hz',22)
        self.rounded(d,[1330,186,1872,966])
        state,caption=STATES.get(cell.state,(cell.state,''))
        self.text(d,(1360,214),'AUTONOMOUS SORTING',22,MUTED)
        self.text(d,(1360,255),state,36)
        done=len(cell.trials)
        self.text(d,(1360,314),f'完了 {done} / 6',28)
        d.rounded_rectangle([1360,362,1842,372],5,fill='#e1e8ea')
        if done: d.rounded_rectangle([1360,362,1360+482*done/6,372],5,fill='#20a28d')
        cur=cell.current
        self.text(d,(1360,401),'AI判定',22,MUTED)
        if cur:
            self.text(d,(1360,438),JA[cur['class']],40,COL[cur['class']])
            for i,label in enumerate(['RED','BLUE','YELLOW']):
                y=505+i*37
                self.text(d,(1360,y),label,20,MUTED)
                d.rectangle([1460,y+10,1740,y+23],fill='#edf1f3')
                d.rectangle([1460,y+10,1460+280*cur['scores'][i],y+23],fill=COL[label])
                self.text(d,(1755,y-2),f"{cur['scores'][i]*100:.1f}%",20)
            self.text(d,(1360,625),f"色特徴距離 {cur['distance']:.3f}  /  上限 {cur['radius']:.3f}",22,MUTED)
        else:
            self.text(d,(1360,445),'カメラ画像を解析中',28)
        if cell.latest_camera is not None:
            camera=Image.fromarray(cell.latest_camera).resize((238,238))
            frame.paste(camera,(1360,690)); d=ImageDraw.Draw(frame)
            self.text(d,(1360,656),'上方RGB-Dカメラ',20,MUTED)
        self.text(d,(1632,695),'回収数',22,MUTED)
        for i,label in enumerate(['RED','BLUE','YELLOW','CHECK']):
            y=742+i*46
            d.ellipse([1632,y+7,1650,y+25],fill=COL[label])
            self.text(d,(1665,y),label,22)
            self.text(d,(1810,y),str(cell.counts[label]),26)
        d.rectangle([0,996,1920,1080],fill=INK)
        self.text(d,(48,1016),caption,30,WHITE)
        return frame

    def write(self,frame):
        self.proc.stdin.write(frame.tobytes()); self.n+=1

    def frame(self,cell):
        im=self.image(cell)
        if cell.state!=self.last_state:
            print(f'video {self.n/30:.1f}s | sim {cell.time:.1f}s | {cell.state}',flush=True)
            if cell.state in ['SCAN','SUCTION','TRANSFER','COMPLETE']:
                im.save(self.root/f'evidence/frame_{self.n:04}.jpg',quality=90)
            self.last_state=cell.state
        # 15 distinct simulation frames/s, duplicated into a 30 fps MP4.
        # Physics advances 16 x 1/240 s between images, preserving 1x time.
        self.write(im); self.write(im)

    def intro(self,cell):
        cell.latest_camera=None
        im=self.image(cell); d=ImageDraw.Draw(im)
        self.rounded(d,[110,290,1215,838],INK)
        self.text(d,(155,330),'色を見て、つかんで、分ける。',52,WHITE)
        self.text(d,(155,427),'課題：6個の色分け回収物を自動で搬送する',32,WHITE)
        self.text(d,(155,492),'1  RGB-D画像から位置と色特徴を取得',28,'#bdd5df')
        self.text(d,(155,550),'2  学習済みAIで分類し、不確かな色は保留',28,'#bdd5df')
        self.text(d,(155,608),'3  逆運動学＋吸着近似で4つの箱へ',28,'#bdd5df')
        self.text(d,(155,714),'独自実装 / 学習あり / 実機は使用しない',26,'#61d5bc')
        for _ in range(8*30): self.write(im)
        im.save(self.root/'evidence/title.jpg',quality=95)

    def outro(self,cell,result):
        im=self.image(cell); d=ImageDraw.Draw(im)
        self.rounded(d,[100,260,1220,890],INK)
        self.text(d,(145,293),'実行結果と、できることの範囲',40,WHITE)
        self.text(d,(145,367),f"搬送成功  {result['successful']} / {result['objects']}",64,'#61d5bc')
        self.text(d,(145,466),'別シードの位置・色の揺らぎでも動作を確認',28,WHITE)
        self.text(d,(145,527),f"既知色テスト {self.metrics['test_samples']} / {self.metrics['test_samples']} 正解",28,WHITE)
        self.text(d,(145,581),'紫の未学習色テスト 30 / 30 を保留',28,WHITE)
        self.text(d,(145,649),'評価範囲：同じ仮想環境・色分けされた立方体',26,'#bdd5df')
        self.text(d,(145,701),'吸着は固定拘束で近似。実物の材質認識・真空圧・',26,'#bdd5df')
        self.text(d,(145,745),'重なり物体の処理や実機での安全性は未検証。',26,'#bdd5df')
        self.text(d,(145,821),'コード・学習データ・結果ログを添付',24,'#61d5bc')
        for _ in range(12*30): self.write(im)
        im.save(self.root/'evidence/results.jpg',quality=95)

    def close(self):
        self.proc.stdin.close(); code=self.proc.wait(); self.err.close()
        if code: raise RuntimeError(f'FFmpeg failed: {code}')
        (self.root/'evidence/video_metrics.json').write_text(json.dumps({'frames':self.n,'fps':30,
            'duration_seconds':self.n/30,'resolution':[1920,1080],
            'speed':'1x simulation time','simulation_render_fps':15,
            'audio':'none; Japanese explanatory captions'},indent=2),encoding='utf-8')
