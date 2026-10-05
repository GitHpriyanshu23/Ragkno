import fs from 'node:fs';
const sr=48000,n=sr*30,L=new Float32Array(n),R=new Float32Array(n);let seed=47;
const noise=()=>{seed=(Math.imul(seed,1664525)+1013904223)|0;return(seed>>>0)/2147483648-1};
function sound(frame,len,fn,gain=.4){const start=Math.round(frame/30*sr);for(let i=0;i<len*sr&&start+i<n;i++){const v=fn(i/sr)*gain;L[start+i]+=v;R[start+i]+=v;}}
// A short, filtered mouse transient with smooth edges; no sustained tone.
function click(f,g=.28){let low=0;let previous=0;sound(f,.036,t=>{low+=.3*(noise()-low);const transient=low-previous*.7;previous=low;const envelope=Math.min(1,t/.0015)*Math.exp(-t*155)*Math.min(1,(.036-t)/.006);return transient*envelope;},g)}
function key(f){let low=0;sound(f,.025,t=>{low+=.2*(noise()-low);return low*Math.min(1,t/.002)*Math.exp(-t*190)*Math.min(1,(.025-t)/.005);},.075)}
function whoosh(f){let lp=0;sound(f,.55,t=>{lp+=.12*(noise()-lp);return lp*Math.sin(Math.PI*t/.55)*Math.sin(2*Math.PI*(420*t+1400*t*t));},.9)}
function chime(f){sound(f,.7,t=>(Math.sin(2*Math.PI*880*t)+.6*Math.sin(2*Math.PI*1320*t))*Math.exp(-t*7),.22)}
for(const f of [103,203,326,516,616,776])whoosh(f);
for(const f of [251,388,489,597,682,873])click(f);
for(let f=349;f<386;f+=3)key(f);
for(const f of [309,460,702])chime(f);
for(let f=257;f<305;f+=7)sound(f,.05,t=>Math.sin(2*Math.PI*(500+(f-257)*15)*t)*Math.exp(-t*80),.08);
const b=Buffer.alloc(44+n*4);b.write('RIFF');b.writeUInt32LE(b.length-8,4);b.write('WAVEfmt ',8);b.writeUInt32LE(16,16);b.writeUInt16LE(1,20);b.writeUInt16LE(2,22);b.writeUInt32LE(sr,24);b.writeUInt32LE(sr*4,28);b.writeUInt16LE(4,32);b.writeUInt16LE(16,34);b.write('data',36);b.writeUInt32LE(n*4,40);for(let i=0;i<n;i++){b.writeInt16LE(Math.round(Math.tanh(L[i])*28000),44+i*4);b.writeInt16LE(Math.round(Math.tanh(R[i])*28000),46+i*4)}fs.writeFileSync('public/media/launch-sfx.wav',b);
