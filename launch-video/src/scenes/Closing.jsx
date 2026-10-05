import React from 'react';
import {Composition,useCurrentFrame} from 'remotion';
import {Stage,Word,tween,pop,Pointer,C} from '../motion';
import {Logo} from '../design';
export function Closing(){const f=useCurrentFrame();return <Stage>
<div style={{position:'absolute',left:0,top:210,width:'100%',display:'flex',alignItems:'center',justifyContent:'center',gap:40,scale:.8+pop(f)*.2}}><div style={{rotate:`${tween(f,0,30,-135,0)}deg`}}><Logo size={160} dark/></div><span style={{fontSize:195,fontWeight:750,letterSpacing:-13}}>RagKno</span></div>
<Word delay={18} style={{position:'absolute',top:485,width:'100%',textAlign:'center',fontFamily:'Times New Roman',fontStyle:'italic',fontSize:78}}>Your knowledge. Now in conversation.</Word>
<div style={{position:'absolute',left:560,top:675,width:800,height:145,display:'flex',alignItems:'center',justifyContent:'center',gap:55,borderRadius:85,background:f>=88?C.blue:C.ink,color:'#fff',fontSize:65,fontWeight:600,letterSpacing:-2,scale:pop(f,32),boxShadow:'0 20px 50px #1112'}}>ragkno.com <span>↗</span></div>
<Pointer x={tween(f,60,87,1510,1260)} y={tween(f,60,87,970,760)} click={f>=88&&f<94}/>
<Word delay={58} style={{position:'absolute',top:875,width:'100%',textAlign:'center',fontSize:28,color:'#777'}}>Start a conversation with your documents.</Word>
</Stage>}
export const ClosingRegistration=()=> <Composition id="Closing" component={Closing} durationInFrames={140} fps={30} width={1920} height={1080}/>;
