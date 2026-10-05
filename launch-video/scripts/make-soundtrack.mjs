import {execFileSync} from 'node:child_process';
execFileSync(process.execPath,['scripts/make-music.mjs'],{stdio:'inherit'});
execFileSync(process.execPath,['scripts/make-sfx.mjs'],{stdio:'inherit'});
execFileSync('ffmpeg',['-hide_banner','-loglevel','error','-y','-i','public/media/launch-music.wav','-i','public/media/launch-sfx.wav','-filter_complex','[0:a]volume=0.9[m];[m][1:a]amix=inputs=2:normalize=0,alimiter=limit=0.92:level=false[a]','-map','[a]','-codec:a','libmp3lame','-b:a','256k','public/media/launch-electronic-soundtrack-v2.mp3'],{stdio:'inherit'});
console.log('Created browser-compatible 30-second electronic + sound effects master.');
