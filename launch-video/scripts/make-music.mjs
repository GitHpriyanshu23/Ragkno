// Original instrumental score: Knowledge in Motion. No sampled recordings.
// 128 BPM × 64 beats = exactly 30 seconds. Deterministic stereo synthesis.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sr=48000, duration=30, beat=60/128, count=sr*duration;
const left=new Float32Array(count), right=new Float32Array(count);
let seed=42;
const noise=()=>{seed=(Math.imul(seed,1664525)+1013904223)|0;return (seed>>>0)/2147483648-1;};
const freq=midi=>440*2**((midi-69)/12);
function sound(start,length,voice,pan=0,gain=1){
  const first=Math.round(start*sr), samples=Math.round(length*sr);
  const l=Math.sqrt((1-pan)/2)*gain,r=Math.sqrt((1+pan)/2)*gain;
  for(let i=0;i<samples && first+i<count;i++){
    if(first+i<0)continue;
    const sample=voice(i/sr,i/samples);
    left[first+i]+=sample*l;right[first+i]+=sample*r;
  }
}
const chords=[[50,53,57,60],[46,50,53,57],[41,45,48,52],[48,52,55,59]];
for(let bar=0;bar<16;bar++){
  const chord=chords[Math.floor(bar/2)%4];
  const start=bar*4*beat;
  // Airy, slowly opening pad. The sidechain leaves space for the drums.
  for(const [j,note] of chord.entries()){
    sound(start,4*beat+.45,(t,p)=>{
      const env=Math.min(1,t/.15)*Math.min(1,(1-p)*10);
      const duck=.55+.45*(1-Math.exp(-((t%beat)/.10)));
      const hz=freq(note+12);
      return env*duck*(Math.sin(2*Math.PI*hz*t)+.22*Math.sin(2*Math.PI*hz*2.002*t))*.047;
    },(j-1.5)/2);
  }
  // Plucked arpeggio with stereo echoes, more movement after the reveal.
  for(let step=0;step<8;step++){
    const note=chord[[0,2,1,3,2,1,3,2][step]]+24;
    const pluck=t=>Math.exp(-t*7)*(Math.sin(2*Math.PI*freq(note)*t)+.22*Math.sin(2*Math.PI*freq(note)*2*t))*.11;
    sound(start+step*beat/2,.65,pluck,step%2?.35:-.35,bar<2?.65:1);
    sound(start+step*beat/2+beat*.75,.65,pluck,step%2?-.7:.7,.24);
  }
  if(bar>=2 && bar<15){
    for(let b=0;b<4;b++){
      sound(start+b*beat,.3,t=>Math.exp(-t*16)*Math.sin(2*Math.PI*(48*t+5*(1-Math.exp(-t*38))))*.55);
      if(b===1 || b===3){
        sound(start+b*beat,.16,t=>noise()*Math.exp(-t*28)*.16+Math.sin(2*Math.PI*185*t)*Math.exp(-t*32)*.12,.05);
        sound(start+b*beat+.012,.12,t=>noise()*Math.exp(-t*38)*.10,-.1);
      }
    }
    for(let s=0;s<8;s++){
      sound(start+s*beat/2,.055,t=>noise()*Math.exp(-t*95)*.07,s%2?.6:-.6,s%2?.7:1);
    }
    // Rounded bass, with rests rather than a continuous drone.
    for(const [offset,len] of [[0,.65],[1.5,.35],[2,.65],[3.5,.35]]){
      const hz=freq(chord[0]-12);
      sound(start+offset*beat,len*beat,(t,p)=>Math.min(1,t/.012)*Math.min(1,(1-p)*12)*(Math.sin(2*Math.PI*hz*t)+.18*Math.sin(2*Math.PI*hz*2*t))*.22);
    }
  }
}
// Soft sweeps and tonal impacts on the authored scene cuts.
for(const at of [3.75,7.5,12.2,20.633333,25.333333]){
  sound(at-.24,.5,(t,p)=>noise()*Math.sin(Math.PI*p)**2*.07,(at%2)-1);
  sound(at,1.4,t=>Math.exp(-t*5)*(Math.sin(2*Math.PI*freq(74)*t)+.4*Math.sin(2*Math.PI*freq(81)*t))*.07);
}
// Light stereo room/delay, followed by controlled saturation and final fade.
for(let i=count-1;i>=Math.round(sr*beat*.75);i--){
  const delayed=i-Math.round(sr*beat*.75);
  left[i]+=right[delayed]*.10;right[i]+=left[delayed]*.10;
}
const wav=Buffer.alloc(44+count*4);
wav.write('RIFF',0);wav.writeUInt32LE(wav.length-8,4);wav.write('WAVEfmt ',8);
wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(2,22);
wav.writeUInt32LE(sr,24);wav.writeUInt32LE(sr*4,28);wav.writeUInt16LE(4,32);wav.writeUInt16LE(16,34);wav.write('data',36);wav.writeUInt32LE(count*4,40);
for(let i=0;i<count;i++){
  const t=i/sr,fade=Math.min(1,t/.08)*Math.min(1,Math.max(0,(30-t)/.8));
  wav.writeInt16LE(Math.round(Math.tanh(left[i]*1.45)*fade*30000),44+i*4);
  wav.writeInt16LE(Math.round(Math.tanh(right[i]*1.45)*fade*30000),46+i*4);
}
fs.mkdirSync(path.join(root,'public/media'),{recursive:true});
fs.writeFileSync(path.join(root,'public/media/launch-music.wav'),wav);
console.log('Created 30-second original stereo score, 48 kHz.');
