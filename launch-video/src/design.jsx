import React from 'react';
import {AbsoluteFill, interpolate, spring, useCurrentFrame} from 'remotion';

export const palette = {cream:'#f7f5f1', ink:'#111111', blue:'#3b82f6', muted:'#a3a3a3', green:'#34d399'};
export const enter = (frame, delay = 0) => spring({frame:frame-delay, fps:30, config:{damping:20, stiffness:140, mass:0.8}});
export const clamp = {extrapolateLeft:'clamp', extrapolateRight:'clamp'};

export function Logo({size=64, dark=false}) {
  return <svg width={size} height={size} viewBox="0 0 100 100" fill="none">
    <rect x="3" y="3" width="94" height="94" rx="24" stroke={dark ? palette.ink : palette.cream} strokeWidth="5"/>
    <path d="M50 22V78M22 50H78M30 30L70 70M70 30L30 70" stroke={dark ? palette.ink : palette.cream} strokeWidth="5"/>
  </svg>;
}
export function Brand({dark=false, size=46}) {
  return <div style={{display:'flex',alignItems:'center',gap:18,fontSize:size,fontWeight:750,letterSpacing:-2,color:dark?palette.ink:palette.cream}}><Logo size={size*1.3} dark={dark}/>RagKno</div>;
}
export function Shell({children, light=false, number, label}) {
  return <AbsoluteFill style={{background:light?palette.cream:palette.ink,color:light?palette.ink:palette.cream,fontFamily:'Arial, Helvetica, sans-serif',overflow:'hidden'}}>
    <div style={{position:'absolute',inset:0,backgroundImage:`radial-gradient(${light?'#11111118':'#ffffff14'} 1px,transparent 1px)`,backgroundSize:'32px 32px',maskImage:'linear-gradient(180deg,transparent 20%,black)',opacity:0.4}}/>
    <div style={{position:'absolute',left:80,top:54}}><Brand dark={light} size={32}/></div>
    <div style={{position:'absolute',right:80,top:68,fontSize:20,letterSpacing:3,color:light?'#676767':'#a3a3a3'}}>DOCUMENTS → ANSWERS</div>
    {children}
    {number && <div style={{position:'absolute',bottom:45,left:80,right:80,display:'flex',justifyContent:'space-between',fontSize:20,letterSpacing:2,color:light?'#676767':'#a3a3a3'}}><span>{number} / 05</span><span>{label}</span><span>RAGKNO.COM</span></div>}
  </AbsoluteFill>;
}
export function Reveal({children, delay=0, style={}}) {
  const frame=useCurrentFrame(); const progress=enter(frame,delay);
  return <div style={{opacity:interpolate(frame,[delay,delay+12],[0,1],clamp),translate:`0 ${interpolate(progress,[0,1],[65,0])}px`,...style}}>{children}</div>;
}
export function Window({children, style={}}) {
  return <div style={{border:'1px solid #ffffff25',borderRadius:24,background:'#070707',overflow:'hidden',boxShadow:'0 35px 90px #00000045',...style}}>
    <div style={{height:42,display:'flex',alignItems:'center',padding:'0 20px',gap:7,background:'#1b1b1b',borderBottom:'1px solid #ffffff15'}}><span style={{width:8,height:8,borderRadius:10,background:'#777'}}/><span style={{width:8,height:8,borderRadius:10,background:'#555'}}/><span style={{width:8,height:8,borderRadius:10,background:'#444'}}/><span style={{position:'absolute',left:'45%',fontSize:15,color:'#aaa'}}>ragkno.com</span></div>
    {children}
  </div>;
}
export function Document({name, type='PDF', style={}}) {
  return <div style={{background:'#ffffff',border:'1px solid #dedbd5',borderRadius:22,padding:'28px 30px',boxShadow:'0 25px 70px #11111112',width:300,...style}}><div style={{display:'flex',alignItems:'center',gap:14,fontSize:19,fontWeight:600}}><svg width="34" height="42" viewBox="0 0 34 42" fill="none"><path d="M6 2H22L32 12V36Q32 40 28 40H6Q2 40 2 36V6Q2 2 6 2Z M22 2V12H32 M9 22H25 M9 29H21" stroke="#111" strokeWidth="2.5"/></svg><span>{type}</span></div><div style={{fontSize:25,fontWeight:700,marginTop:24}}>{name}</div><div style={{height:6,background:'#e4e1db',marginTop:22,borderRadius:4}}/><div style={{height:6,width:'75%',background:'#e4e1db',marginTop:10,borderRadius:4}}/></div>;
}
