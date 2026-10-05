// Open Road: an original, synthesized rock instrumental. No external samples.
import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
const sr=48000,N=sr*30,beat=60/128;
const L=new Float32Array(N),R=new Float32Array(N);
let seed=93;
const noise=()=>{seed=(Math.imul(seed,1664525)+1013904223)|0;return(seed>>>0)/2147483648-1;};
const hz=n=>440*2**((n-69)/12);
function put(at,len,voice,pan=0,gain=1){const start=Math.round(at*sr),n=Math.floor(len*sr);for(let i=0;i<n&&start+i<N;i++){if(start+i<0)continue;const v=voice(i/sr,i/n)*gain;L[start+i]+=v*Math.sqrt((1-pan)/2);R[start+i]+=v*Math.sqrt((1+pan)/2);}}
function guitar(at,note,len,pan,amp){
 const delay=Math.round(sr/hz(note)),ring=Float32Array.from({length:delay},()=>noise());let cursor=0,lp=0;
 put(at,len,(t,p)=>{const a=ring[cursor],b=ring[(cursor+1)%delay];ring[cursor]=(a+b)*.499;cursor=(cursor+1)%delay;const driven=Math.tanh(a*7);lp+=.28*(driven-lp);return lp*Math.min(1,t/.002)*Math.min(1,(1-p)*14)*amp;},pan);
}
function crash(at,gain=.12){let last=0;put(at,1.4,t=>{const n=noise(),high=n-last;last=n;return(high*.7+Math.sin(2*Math.PI*7343*t)*.12)*Math.exp(-t*4)*gain;},.35);}
for(let bar=0;bar<16;bar++){
 const root=[40,43,45,48][Math.floor(bar/2)%4],start=bar*4*beat;
 const hits=bar<2?[0,1.5,2,3.5]:[0,.5,1,1.5,2,2.5,3,3.5];
 for(const [j,b]of hits.entries()){
  const length=(j%4===0?.8:.38)*beat;
  for(const side of [-.78,.78])for(const [k,n]of [root,root+7,root+12].entries())guitar(start+b*beat+(side>0?.009:0)+k*.002,n,length,side,.115);
  put(start+b*beat,length,(t,p)=>Math.tanh((Math.sin(2*Math.PI*hz(root-12)*t)+.2*Math.sin(2*Math.PI*hz(root)*t))*1.8)*.17*Math.min(1,t/.006)*Math.min(1,(1-p)*8));
 }
 if(bar>=2&&bar<15){
  for(const b of [0,1.5,2,2.75])put(start+b*beat,.32,t=>Math.sin(2*Math.PI*(48*t+4.5*(1-Math.exp(-t*42))))*Math.exp(-t*15)*.55);
  for(const b of [1,3]){let last=0;put(start+b*beat,.23,t=>{const n=noise();const high=n-last*.75;last=n;return(high*.23+Math.sin(2*Math.PI*185*t)*.23)*Math.exp(-t*19);},-.08);}
  for(let h=0;h<8;h++){let last=0;put(start+h*beat/2,.08,t=>{const n=noise(),high=n-last;last=n;return high*Math.exp(-t*70)*.055;},.4,h%2?.7:1);}
 }
 if([0,2,4,8,12,14].includes(bar))crash(start);
 // A short pentatonic lead answers the rhythm guitar during the second half.
 if(bar>=8&&bar<15)for(const [j,n]of [root+24,root+27,root+31,root+29].entries())guitar(start+(j+.5)*beat,n,.42,-.15,.10);
 if(bar===7||bar===13)for(let j=0;j<4;j++)put(start+(3+j/4)*beat,.18,t=>Math.sin(2*Math.PI*(125-j*13)*t)*Math.exp(-t*20)*.23+noise()*Math.exp(-t*35)*.06,j/4-.4);
}
// Resolve into a held E power chord for the end card.
for(const side of [-.8,.8])for(const note of [40,47,52])guitar(28.125,note,1.875,side,.14);
crash(28.125,.15);
let peak=0;for(let i=0;i<N;i++){L[i]=Math.tanh(L[i]*1.7);R[i]=Math.tanh(R[i]*1.7);peak=Math.max(peak,Math.abs(L[i]),Math.abs(R[i]));}
const wav=Buffer.alloc(44+N*4);wav.write('RIFF');wav.writeUInt32LE(wav.length-8,4);wav.write('WAVEfmt ',8);wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(2,22);wav.writeUInt32LE(sr,24);wav.writeUInt32LE(sr*4,28);wav.writeUInt16LE(4,32);wav.writeUInt16LE(16,34);wav.write('data',36);wav.writeUInt32LE(N*4,40);
for(let i=0;i<N;i++){const t=i/sr,fade=Math.min(1,t/.015)*Math.min(1,(30-t)/.4);wav.writeInt16LE(Math.round(L[i]/peak*.86*fade*32767),44+i*4);wav.writeInt16LE(Math.round(R[i]/peak*.86*fade*32767),46+i*4);}
fs.writeFileSync(fileURLToPath(new URL('../public/media/launch-rock.wav',import.meta.url)),wav);console.log('Created Open Road: 30 seconds, 128 BPM, stereo rock.');
