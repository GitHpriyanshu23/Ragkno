import React from 'react';
import {useCurrentFrame} from 'remotion';
import {tween,pop} from './motion';
export function Details(){const f=useCurrentFrame();const processing=f>=212&&f<335;const querying=f>=335&&f<525;if(!processing&&!querying)return null;const local=f-(processing?212:335);return <div style={{position:'absolute',left:80,top:110,display:'flex',gap:14,pointerEvents:'none'}}>{(processing?['01 CONNECT','02 UPLOAD → INDEX']:['03 ASK','04 ANSWER + CITATIONS']).map((t,i)=><div key={t} style={{fontFamily:'Arial',fontSize:18,letterSpacing:2,padding:'12px 18px',borderRadius:25,color:processing?'#2563eb':'#b7cfff',border:'1px solid #3b82f655',background:processing?'#eaf1fe':'#1b2940',opacity:tween(local,i*7,i*7+10,0,1),scale:pop(local,i*7)}}>{t}</div>)}</div>}
