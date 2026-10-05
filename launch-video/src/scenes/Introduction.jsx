import React from 'react';
import {Composition,useCurrentFrame} from 'remotion';
import {Stage,Paper,tween,pop,C} from '../motion';
import {Logo} from '../design';
export function Introduction(){const f=useCurrentFrame()*1.13;const collapse=tween(f,34,65,0,1);return <Stage>
<div style={{position:'absolute',inset:0,opacity:1-collapse}}>{[{x:120,y:135,r:-14,t:'Your PDFs',a:C.blue},{x:1350,y:110,r:12,t:'Drive files',a:'#2bac80'},{x:320,y:660,r:8,t:'Research notes',a:'#d4a346'},{x:1300,y:650,r:-9,t:'Web pages',a:'#9b7cdc'}].map((p,i)=><Paper key={p.t} title={p.t} accent={p.a} style={{position:'absolute',left:p.x+(960-p.x-165)*collapse,top:p.y+(540-p.y-205)*collapse,rotate:`${p.r*(1-collapse)}deg`,scale:1-collapse*.65,translate:`0 ${Math.sin(f/18+i)*12}px`}}/>)}</div>
<div style={{position:'absolute',left:0,top:350,width:'100%',textAlign:'center',opacity:1-tween(f,28,45,0,1),fontSize:110,fontWeight:700,letterSpacing:-6}}>All that knowledge.<br/><span style={{fontFamily:'Times New Roman',fontWeight:400,fontStyle:'italic'}}>One place to ask.</span></div>
<div style={{position:'absolute',left:860,top:390,width:200,height:200,borderRadius:tween(f,60,94,100,48),background:C.ink,display:'flex',alignItems:'center',justifyContent:'center',scale:pop(f,47),rotate:`${tween(f,47,85,-90,0)}deg`,boxShadow:'0 20px 70px #1112'}}><Logo size={125}/></div>
<div style={{position:'absolute',top:640,width:'100%',textAlign:'center',fontSize:104,fontWeight:750,letterSpacing:-6,opacity:tween(f,65,80,0,1),translate:`0 ${(1-pop(f,66))*45}px`}}>Meet RagKno.</div><div style={{position:"absolute",top:795,width:"100%",textAlign:"center",fontSize:35,opacity:tween(f,76,87,0,1)}}>AI answers from your PDFs, Drive files and web pages.</div>
</Stage>}
export const IntroductionRegistration=()=> <Composition id="Introduction" component={Introduction} durationInFrames={113} fps={30} width={1920} height={1080}/>;
