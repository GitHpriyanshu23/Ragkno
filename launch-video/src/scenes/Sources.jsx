import React from 'react';
import {Composition,useCurrentFrame} from 'remotion';
import {Stage,Paper,Pointer,Word,tween,pop,C} from '../motion';
export function Sources(){const f=useCurrentFrame()*1.15;const dropped=f>=46;const progress=tween(f,48,108,0,100);return <Stage>
<Word style={{position:'absolute',top:155,width:'100%',textAlign:'center',fontSize:94,fontWeight:700,letterSpacing:-5}}>Connect your files. <span style={{color:C.blue}}>Ask your AI.</span></Word>
<div style={{position:'absolute',top:295,left:500,display:'flex',alignItems:'center',gap:35}}>{['Upload','Extract text','Searchable knowledge'].map((text,i)=><React.Fragment key={text}>{i>0&&<span style={{fontSize:30,color:'#3b82f6'}}>→</span>}<div style={{padding:'12px 22px',borderRadius:30,background:f>48+i*24?'#111':'#e4e1db',color:f>48+i*24?'#fff':'#888',fontSize:25,scale:.95+pop(f,48+i*24)*.05}}>{f>75+i*18?'✓ ':''}{text}</div></React.Fragment>)}</div>
<div style={{position:'absolute',left:390,top:370,width:1140,height:460,border:`2px dashed ${dropped?'#3b82f6':'#bbb8b1'}`,borderRadius:35,background:dropped?'#eaf1fe':'#eeece7',scale:.96+pop(f)*.04}}>
<div style={{position:'absolute',inset:0,display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',opacity:1-tween(f,43,52,0,1)}}><div style={{fontSize:90}}>↓</div><div style={{fontSize:38,fontWeight:600}}>Your next answer starts here</div><div style={{fontSize:24,color:'#777',marginTop:20}}>PDF · Google Drive · Web pages</div></div>
<div style={{position:'absolute',left:65,right:65,top:108,opacity:tween(f,49,60,0,1),translate:`0 ${(1-pop(f,50))*30}px`,padding:38,background:'#fff',borderRadius:25,boxShadow:'0 20px 55px #152d5520'}}><div style={{display:'flex',gap:25,alignItems:'center'}}><div style={{background:'#edf3fe',borderRadius:14,padding:20,fontSize:28}}>PDF</div><div><div style={{fontSize:37,fontWeight:700}}>Annual report.pdf</div><div style={{fontSize:25,color:'#777',marginTop:9}}>{f<77?'Uploading document…':f<111?'Indexing your knowledge…':'Ready for your questions'}</div></div><span style={{marginLeft:'auto',fontSize:34,color:f<111?C.blue:'#20a27a'}}>{f<111?`${Math.floor(progress)}%`:'✓'}</span></div><div style={{height:9,borderRadius:8,background:'#eee',marginTop:30,overflow:'hidden'}}><div style={{height:'100%',width:`${progress}%`,background:f<111?C.blue:'#20a27a'}}/></div></div>
</div>
<Paper title="Annual report" style={{position:'absolute',left:tween(f,9,45,100,740),top:tween(f,9,45,565,360),rotate:`${tween(f,9,45,-13,0)}deg`,scale:tween(f,40,53,.7,.2),opacity:1-tween(f,44,54,0,1)}}/>
<Pointer x={tween(f,9,45,365,990)} y={tween(f,9,45,760,570)} click={f>=44&&f<49}/>
<Word delay={111} style={{position:'absolute',top:900,width:'100%',textAlign:'center',fontSize:32,color:'#20a27a'}}>Indexed for search. Ready for AI answers.</Word>
</Stage>}
export const SourcesRegistration=()=> <Composition id="Sources" component={Sources} durationInFrames={141} fps={30} width={1920} height={1080}/>;
