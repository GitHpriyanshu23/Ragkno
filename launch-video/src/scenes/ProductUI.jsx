import React from 'react';
import {Video} from '@remotion/media';
import {Composition,staticFile,useCurrentFrame} from 'remotion';
import {Stage,tween,pop,Pointer} from '../motion';
export function ProductUI(){const f=useCurrentFrame();return <Stage dark>
<div style={{position:'absolute',inset:0,background:'radial-gradient(ellipse at center,#3b82f640,transparent 70%)'}}/>
<div style={{position:'absolute',left:170,top:150,width:1580,height:790,border:'1px solid #ffffff55',borderRadius:24,overflow:'hidden',boxShadow:'0 40px 100px #0008',transform:`perspective(1600px) rotateX(${tween(f,0,28,12,0)}deg) rotateY(${tween(f,0,35,-8,0)}deg) scale(${.85+pop(f)*.15})`}}>
<div style={{height:44,background:'#222',display:'flex',alignItems:'center',gap:9,padding:'0 22px'}}>{['#f87171','#facc15','#4ade80'].map(c=><span key={c} style={{width:10,height:10,borderRadius:10,background:c}}/>)}<span style={{marginLeft:570,fontSize:18,color:'#bbb'}}>ragkno.com</span></div>
<Video
  src={staticFile('media/home.mp4')}
  muted
  style={{
    width: 1580,
    height: 946,
    transform: `translateY(${tween(f, 45, 95, 0, -45)}px) scale(${tween(f, 25, 90, 1, 1.04)})`,
    transformOrigin: 'center top',
    backgroundColor: 'rgba(255, 255, 255, 0)'
  }}
/>
</div>
<Pointer x={tween(f,40,70,1580,866)} y={tween(f,40,70,930,625)} click={f>=72&&f<78}/>
</Stage>}
export const ProductUIRegistration=()=> <Composition id="ProductUI" component={ProductUI} durationInFrames={100} fps={30} width={1920} height={1080}/>;
