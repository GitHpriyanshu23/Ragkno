import React from 'react';
import {Composition, interpolate, useCurrentFrame} from 'remotion';
import {Shell, Document, Reveal} from '../design';

export function Opening() {
  const f=useCurrentFrame();
  return <Shell light>
    <Document name="Quarterly report" style={{position:'absolute',left:95,top:175,rotate:`${interpolate(f,[0,112],[-14,-9])}deg`,translate:`0 ${interpolate(f,[0,112],[45,-10])}px`,opacity:0.65}}/>
    <Document name="Research notes" type="DOCX" style={{position:'absolute',right:85,top:150,rotate:'12deg',translate:`0 ${interpolate(f,[0,112],[-15,25])}px`,opacity:0.65}}/>
    <Document name="Knowledge base" type="DRIVE" style={{position:'absolute',left:230,bottom:-100,rotate:'-7deg',opacity:0.65}}/>
    <div style={{position:'absolute',left:0,right:0,top:350,textAlign:'center',fontSize:150,lineHeight:1.06,fontWeight:750,letterSpacing:-8}}>
      <Reveal delay={0}>Stop searching.</Reveal>
      <Reveal delay={20} style={{color:'#3b82f6'}}>Start asking.</Reveal>
    </div>
    <Reveal delay={48} style={{position:'absolute',top:730,width:'100%',textAlign:'center',fontSize:34,color:'#676767'}}>Your documents have the answers.</Reveal>
  </Shell>;
}
export const OpeningRegistration=()=> <Composition id="Opening" component={Opening} durationInFrames={112} fps={30} width={1920} height={1080} defaultProps={{}}/>;
