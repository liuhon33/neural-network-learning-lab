
(()=>{
const root=document.getElementById('rs-lab'),$=s=>root.querySelector(s),NS='http://www.w3.org/2000/svg';
/*__ENGINE__*/

const make=(tag,attrs,text)=>{const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e};
let nets,points,step=0,running=false,surfaceCache=null,featureCache=null,mseCache=null;
const inputPoint={x:.5,y:-.5};
let widths=Array(37).fill(10),order=[],batchCursor=0,samplesSeen=0,batchRng,trainTimer=null;
const pcaPrevious=new Map(),correctionPcaPrevious=new Map(),batchSize=128;
const camera={yaw:-.72,pitch:.58},surfaceSize=25;
const surfaceGrid=Array.from({length:surfaceSize*surfaceSize},(_,k)=>[-2+4*(k%surfaceSize)/(surfaceSize-1),-2+4*Math.floor(k/surfaceSize)/(surfaceSize-1)]);
const gridSide=17,grid=[];for(let j=0;j<gridSide;j++)for(let i=0;i<gridSide;i++)grid.push([-2+4*i/(gridSide-1),-2+4*j/(gridSide-1)]);
let runStart=0,runGoal=0;
function updateTrainingUI(){
 for(const id of ['rs-run','rs-run-500','rs-run-1000','rs-step'])$('#'+id).disabled=running;
 $('#rs-pause').disabled=!running;
 for(const control of $('#rs-configuration-panel').querySelectorAll('input,select,button'))control.disabled=running;
 $('#rs-progress').max=Math.max(1,runGoal);$('#rs-progress').value=Math.max(0,step-runStart);
 if(running)$('#rs-run-status').textContent=(step-runStart).toLocaleString()+' / '+runGoal.toLocaleString()+' additional updates · total '+step.toLocaleString()+'. Pause keeps your progress.';
}
function stop(message){running=false;clearTimeout(trainTimer);trainTimer=null;if(message)$('#rs-run-status').textContent=message;updateTrainingUI()}
function shuffle(){for(let i=order.length-1;i>0;i--){const j=Math.floor(batchRng()*(i+1));[order[i],order[j]]=[order[j],order[i]]}}
function buildWidthControls(){
 const box=$('#rs-width-controls');box.replaceChildren();
 widths.forEach((w,i)=>{
  const div=document.createElement('div');div.className='rs-control';
  div.innerHTML='<div class="rs-control-heading"><label for="rs-width-'+i+'">'+(i===0?'Stem':'Block '+i)+'</label><input id="rs-width-number-'+i+'" class="form-control rs-number" type="number" min="1" max="32" value="'+w+'" aria-label="'+(i===0?'Stem':'Block '+i)+' width numeric value"></div><input id="rs-width-'+i+'" class="form-range" type="range" min="1" max="32" value="'+w+'" aria-label="'+(i===0?'Stem':'Block '+i)+' width">';
  box.append(div);
  $('#rs-width-'+i).oninput=()=>{widths[i]=Number($('#rs-width-'+i).value);$('#rs-width-number-'+i).value=widths[i];init()};
  $('#rs-width-number-'+i).onchange=()=>{const el=$('#rs-width-number-'+i);widths[i]=Math.max(1,Math.min(32,Math.round(Number(el.value)||1)));el.value=widths[i];$('#rs-width-'+i).value=widths[i];init()};
 });
}
function init(){
 stop();runStart=0;runGoal=0;$('#rs-run-status').textContent='Ready. Each update uses a batch of up to 128 examples.';const depth=widths.length-1,seed=Math.max(1,Math.min(999,Math.round(Number($('#rs-seed').value)||7)));$('#rs-seed').value=seed;
 nets=[new Net(widths,false,seed),new Net(widths,true,seed)];points=trainingPoints(Number($('#rs-samples').value));
 step=0;samplesSeen=0;batchCursor=0;order=points.map((_,i)=>i);batchRng=rng(seed^0x2a7f19);shuffle();
 surfaceCache=featureCache=mseCache=null;pcaPrevious.clear();correctionPcaPrevious.clear();$('#rs-layer').max=depth;$('#rs-layer-number').max=depth;$('#rs-layer').value=Math.min(Number($('#rs-layer').value),depth);
 const same=widths.every(w=>w===widths[0]);$('#rs-width-mixed').textContent=same?'':'(mixed widths)';
 if(same){$('#rs-width-all').value=widths[0];$('#rs-width-all-number').value=widths[0]}
 $('#rs-architecture').textContent='Input 2 → stem '+widths[0]+widths.slice(1).map((w,i)=>' → block '+(i+1)+' ['+w+']').join('')+' → output 1';
 const projections=nets[1].layers.filter(z=>z.skip),extra=projections.reduce((s,z)=>s+z.skip.length,0);
 $('#rs-shortcuts').textContent=extra?projections.length+' width-changing shortcut(s): learned linear maps, '+extra+' extra residual parameters. Equal-width shortcuts remain identities. Shrinking a layer creates a bottleneck.':'All residual shortcuts are identities; both networks have the same parameter count.';
 $('#rs-target-surface').setAttribute('aria-label','Target function with '+points.length+' training samples');inspectorReset();updateTrainingUI();draw();
}
function featureData(layer){
 if(featureCache?.owner===nets&&featureCache.step===step&&featureCache.layer===layer)return featureCache;
 const forwards=nets.map(n=>grid.map(p=>n.forward(...p,layer))),raw=forwards.map(rows=>rows.map(c=>Array.from(c.hs[layer+1]))),dim=widths[layer];
 const pca=dim>2?fitPCA(raw.flat(),pcaPrevious.get(layer)):null;if(pca)pcaPrevious.set(layer,pca);
 const project=pca?h=>projectPCA(h,pca):h=>[h[0],h[1]||0];
 let correction=null;
 if(layer>0){
  // Use the actual branch activation, including the residual scale. A width-changing
  // block adds this vector to P h, so output minus the unprojected input is invalid.
  const raw=forwards[1].map(c=>Array.from(c.acts[layer],v=>nets[1].alpha*v));
  const pca=dim>2?fitPCA(raw,correctionPcaPrevious.get(layer)):null;if(pca)correctionPcaPrevious.set(layer,pca);
  const project=pca?h=>projectPCA(h,pca):h=>[h[0],h[1]||0];
  correction={raw,dim,pca,project,cloud:raw.map(project),zero:project(new Float64Array(dim)),retained:pca?pca.retained:1};
 }
 featureCache={owner:nets,step,layer,raw,dim,pca,project,correction,clouds:raw.map(rows=>rows.map(project)),retained:pca?raw.map(rows=>retainedVariance(rows,pca)):[1,1]};return featureCache;
}
function plot(svg,values,selected,extent,color,labels,reference=null){svg.setAttribute('viewBox','0 0 250 235');svg.replaceChildren();const X=v=>125+v/extent*90,Y=v=>111-v/extent*90,precision=extent<.2?3:extent<2?2:1;for(let k=-2;k<=2;k++){let v=k*extent/2;const tick=precision>1?String(Number(v.toFixed(precision))):v.toFixed(1);svg.append(make('path',{d:'M35,'+Y(v)+'H215 M'+X(v)+',21V201',stroke:'var(--border)','stroke-width':1,fill:'none'}));svg.append(make('text',{x:X(v),y:219,'text-anchor':'middle',class:'text-small'},tick));if(k!==0)svg.append(make('text',{x:precision>1?34:30,y:Y(v)+4,'text-anchor':'end',class:'text-small'},tick))}
for(let j=0;j<gridSide;j++){let d='';for(let i=0;i<gridSide;i++){const p=values[j*gridSide+i];d+=(i?'L':'M')+X(p[0])+','+Y(p[1])}svg.append(make('path',{d,stroke:color,opacity:.35,fill:'none','stroke-width':1}))}
for(let i=0;i<gridSide;i++){let d='';for(let j=0;j<gridSide;j++){const p=values[j*gridSide+i];d+=(j?'L':'M')+X(p[0])+','+Y(p[1])}svg.append(make('path',{d,stroke:color,opacity:.35,fill:'none','stroke-width':1}))}
values.forEach(p=>svg.append(make('circle',{cx:X(p[0]),cy:Y(p[1]),r:1.1,fill:color})));if(reference){const x=X(reference[0]),y=Y(reference[1]);svg.append(make('path',{d:'M'+(x-4)+','+y+'h8 M'+x+','+(y-4)+'v8',stroke:'var(--foreground)','stroke-width':1.5,fill:'none','data-reference':'zero correction'}))}if(svg.id==='rs-input')svg.append(make('circle',{cx:X(selected[0]),cy:Y(selected[1]),r:20,fill:'transparent','data-input-handle':'true'}));svg.append(make('circle',{cx:X(selected[0]),cy:Y(selected[1]),r:svg.id==='rs-input'?6:5,fill:color,stroke:'var(--foreground)','stroke-width':1,class:'rs-selected-point'}));svg.append(make('text',{x:125,y:234,'text-anchor':'middle',class:'text-small'},labels[0]+' →'));svg.append(make('text',{x:35,y:13,class:'text-small'},labels[1]+' ↑'))}
function drawCorrection(view,selected,layer){
 const svg=$('#rs-correction'),caption=$('#rs-correction-caption'),detail=$('#rs-correction-selected');
 svg.dataset.update=step;svg.dataset.block=layer;
 if(!view){
  $('#rs-correction-title').textContent='Stem · no residual branch';svg.setAttribute('viewBox','0 0 250 235');svg.replaceChildren();
  svg.append(make('text',{x:125,y:108,'text-anchor':'middle'},'No residual branch'),make('text',{x:125,y:128,'text-anchor':'middle'},'Select block 1–'+nets[1].depth));
  caption.textContent='The stem transforms the input directly. Added corrections begin at block 1.';detail.textContent='';delete svg.dataset.extent;delete svg.dataset.selectedNorm;return;
 }
 const raw=Array.from(selected.acts[layer],v=>nets[1].alpha*v),marker=view.project(raw);
 const largest=Math.max(...view.cloud.flat().map(Math.abs),...marker.map(Math.abs),...view.zero.map(Math.abs));
 const extent=Math.max(.1,Math.ceil(largest*1.1*20)/20),norm=Math.hypot(...raw);
 $('#rs-correction-title').textContent='Block '+layer+' · added correction αF(h)';
 plot(svg,view.cloud,marker,extent,'var(--viz-series-4)',view.pca?['residual PC1','residual PC2']:['correction 1',view.dim===1?'zero (width 1)':'correction 2'],view.zero);
 svg.dataset.extent=extent;svg.dataset.selectedNorm=norm;
 caption.textContent=(view.pca?view.dim+' correction features → 2 PCs · '+(view.retained===null?'no centered variance':(view.retained*100).toFixed(1)+'% variance retained')+'. Own centered PCA, refitted after every update; its axes differ from the hidden plots. ':view.dim+' correction feature'+(view.dim===1?' (shown on a line).':'s, shown directly.')+' ')+ 'Own scale ±'+extent.toFixed(2)+'. Cross = zero correction. Same probe inputs and grid connections.';
 detail.textContent='Selected αF(h) = ['+raw.map(v=>v.toFixed(4)).join(', ')+']. Full-vector length '+norm.toFixed(4)+' · α = '+nets[1].alpha.toFixed(4)+'.';
}
function surfaceData(){
 if(surfaceCache&&surfaceCache.owner===nets&&surfaceCache.step===step)return surfaceCache;
 const values=[surfaceGrid.map(([x,y])=>target(x,y)),...nets.map(n=>surfaceGrid.map(([x,y])=>n.forward(x,y).pred))];
 const all=[...values.flat(),...points.map(p=>p[2])],low=Math.floor((Math.min(0,...all)-.05)*2)/2,high=Math.ceil((Math.max(0,...all)+.05)*2)/2;
 surfaceCache={owner:nets,step,values,low,high};return surfaceCache;
}
function paintSurface(canvas,values,data,selected,samples){
 const w=canvas.clientWidth,h=canvas.clientHeight;if(w<1||h<1)return;
 const dpr=Math.min(devicePixelRatio||1,2),pw=Math.round(w*dpr),ph=Math.round(h*dpr);
 if(canvas.width!==pw||canvas.height!==ph){canvas.width=pw;canvas.height=ph}
 const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 const style=getComputedStyle(canvas),ink=getComputedStyle(root).color,base=style.color,background=style.backgroundColor,span=data.high-data.low;
 const ca=Math.cos(camera.yaw),sa=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch);
 const raw=([x,y,z])=>{const a=x/2,b=y/2,c=(z-data.low)/span*1.7-.85,depth=a*sa+b*ca;return[a*ca-b*sa,depth*sp-c*cp,depth*cp+c*sp]};
 const corners=[];for(const x of [-2,2])for(const y of [-2,2])for(const z of [data.low,data.high])corners.push(raw([x,y,z]));
 const xs=corners.map(p=>p[0]),ys=corners.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 const scale=Math.min((w-92)/(xmax-xmin),(h-78)/(ymax-ymin)),ox=w/2-(xmin+xmax)*scale/2,oy=h/2-(ymin+ymax)*scale/2-4;
 const project=p=>{const q=raw(p);return[ox+scale*q[0],oy+scale*q[1],q[2]]};
 const path=coords=>{ctx.beginPath();coords.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]))};
 const line=(a,b,alpha=1)=>{ctx.globalAlpha=alpha;ctx.strokeStyle=ink;ctx.lineWidth=1;path([project(a),project(b)]);ctx.stroke();ctx.globalAlpha=1};
 const text=(value,x,y,align='center')=>{ctx.globalAlpha=1;ctx.fillStyle=ink;ctx.font='12px system-ui, sans-serif';ctx.textAlign=align;ctx.textBaseline='middle';ctx.fillText(value,x,y)};
 ctx.fillStyle=background;ctx.fillRect(0,0,w,h);
 for(let k=-2;k<=2;k++){line([-2,k,data.low],[2,k,data.low],.18);line([k,-2,data.low],[k,2,data.low],.18)}
 const projected=surfaceGrid.map(([x,y],k)=>project([x,y,values[k]])),objects=[];
 for(let j=0;j<surfaceSize-1;j++)for(let i=0;i<surfaceSize-1;i++){
  const k=j*surfaceSize+i,ids=[k,k+1,k+1+surfaceSize,k+surfaceSize],vs=ids.map(q=>projected[q]);
  objects.push({kind:'face',coords:vs,depth:vs.reduce((s,p)=>s+p[2],0)/4,z:ids.reduce((s,q)=>s+values[q],0)/4});
 }
 if(samples)for(const p of points){const q=project(p);objects.push({kind:'point',coords:q,depth:q[2]+.018})}
 objects.sort((a,b)=>a.depth-b.depth);
 for(const o of objects){
  if(o.kind==='face'){
   path(o.coords);ctx.closePath();ctx.globalAlpha=1;ctx.fillStyle=background;ctx.fill();
   ctx.globalAlpha=.25+.6*Math.max(0,Math.min(1,(o.z-data.low)/span));ctx.fillStyle=base;ctx.fill();
   ctx.globalAlpha=.28;ctx.lineWidth=.6;ctx.strokeStyle=base;ctx.stroke();
  }else{
   ctx.globalAlpha=.8;ctx.beginPath();ctx.arc(o.coords[0],o.coords[1],Math.max(.65,2.5*Math.sqrt(128/points.length)),0,Math.PI*2);ctx.fillStyle=ink;ctx.fill();ctx.strokeStyle=background;ctx.lineWidth=.35;ctx.stroke();
  }
 }
 ctx.globalAlpha=1;
 const floorCorners=[[-2,-2],[2,-2],[2,2],[-2,2]],near=floorCorners.reduce((best,p)=>raw([...p,data.low])[2]>raw([...best,data.low])[2]?p:best),zCorner=floorCorners.reduce((best,p)=>project([...p,data.low])[0]<project([...best,data.low])[0]?p:best);
 const floorCenter=project([0,0,data.low]);
 for(const [name,point]of [['x₁',v=>[v,near[1],data.low]],['x₂',v=>[near[0],v,data.low]]]){
  const a=project(point(-2)),b=project(point(2)),mid=project(point(0)),dx=b[0]-a[0],dy=b[1]-a[1],length=Math.hypot(dx,dy);
  let nx=-dy/length,ny=dx/length;if(nx*(mid[0]-floorCenter[0])+ny*(mid[1]-floorCenter[1])<0){nx=-nx;ny=-ny}
  line(point(-2),point(2),.65);for(const v of [-2,0,2]){const tick=project(point(v));text(String(v),tick[0]+nx*15,tick[1]+ny*15)}
  text(name,mid[0]+nx*33,mid[1]+ny*33);
 }
 let p;
 line([...zCorner,data.low],[...zCorner,data.high],.65);
 for(const z of [data.low,(data.low+data.high)/2,data.high]){p=project([...zCorner,z]);text(z.toFixed(1),p[0]-7,p[1],'right')}
 p=project([...zCorner,data.high]);text('output z',Math.max(30,p[0]),p[1]-16);
 const marker=project(selected),foot=project([selected[0],selected[1],data.low]);
 ctx.setLineDash([3,3]);path([foot,marker]);ctx.strokeStyle=ink;ctx.globalAlpha=.6;ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
 ctx.beginPath();ctx.arc(marker[0],marker[1],5.5,0,2*Math.PI);ctx.fillStyle=base;ctx.fill();ctx.strokeStyle=ink;ctx.lineWidth=2;ctx.stroke();
 canvas.dataset.update=String(step);canvas.dataset.zMin=data.low;canvas.dataset.zMax=data.high;canvas.dataset.selectedOutput=selected[2];
}
function drawSurfaces(){
 if(!nets)return;const data=surfaceData(),{x,y}=inputPoint,selected=[target(x,y),...nets.map(n=>n.forward(x,y).pred)];
 ['target','plain','res'].forEach((name,i)=>paintSurface($('#rs-'+name+'-surface'),data.values[i],data,[x,y,selected[i]],i===0));
 $('#rs-target-summary').textContent=points.length.toLocaleString()+' training points · selected target '+selected[0].toFixed(4);
 $('#rs-plain-summary').textContent='Selected prediction '+selected[1].toFixed(4)+' · error '+(selected[1]-selected[0]).toFixed(4);
 $('#rs-res-summary').textContent='Selected prediction '+selected[2].toFixed(4)+' · error '+(selected[2]-selected[0]).toFixed(4);
 $('#rs-surface-range').textContent='Shared 3D axes: x₁ and x₂ from −2 to 2; output z from '+data.low.toFixed(1)+' to '+data.high.toFixed(1)+'. The dots on the target are the actual training samples. Prediction surfaces always show the final output; the block slider changes only the hidden-feature plots and highlighted table row.';
}
function draw(){const {x,y}=inputPoint,layer=Number($('#rs-layer').value),depth=nets[0].depth;$('#rs-selected-input').textContent='x₁ = '+x.toFixed(2)+' · x₂ = '+y.toFixed(2);$('#rs-layer-number').value=layer;$('#rs-layer-value').textContent=layer===0?'stem':'block '+layer;
for(const el of root.querySelectorAll('input[type=range]'))el.style.setProperty('--rs-fill',100*(Number(el.value)-Number(el.min))/(Number(el.max)-Number(el.min))+'%');
const selected=nets.map(n=>n.forward(x,y)),view=featureData(layer),clouds=view.clouds,markers=selected.map(c=>view.project(c.hs[layer+1])),extent=Math.max(2,Math.ceil(Math.max(...clouds.flat(2).map(Math.abs),...markers.flat().map(Math.abs))*2)/2),labels=view.pca?['PC1','PC2']:['hidden coordinate 1',view.dim===1?'zero (width 1)':'hidden coordinate 2'];
plot($('#rs-input'),grid,[x,y],2,'var(--foreground)',['input x₁','input x₂']);plot($('#rs-plain'),clouds[0],markers[0],extent,'var(--viz-series-1)',labels);plot($('#rs-res'),clouds[1],markers[1],extent,'var(--viz-series-2)',labels);
drawCorrection(view.correction,selected[1],layer);
$('#rs-info').textContent='Parameters: plain '+nets[0].parameterCount().toLocaleString()+' · residual '+nets[1].parameterCount().toLocaleString()+' · update '+step;
$('#rs-range').textContent=(view.pca?'Live PCA fitted jointly to both networks; recomputed after every update. Signs are aligned, but PCA axes can still rotate as training changes the features. ':'Direct hidden coordinates; no PCA needed. ')+ 'The two hidden-feature plots share axes ±'+extent+'. The correction plot uses its own axes and scale. Connected points are neighbors in the input grid.';
$('#rs-input-caption').textContent=grid.length+' fixed probe inputs for the feature plots; '+points.length.toLocaleString()+' separate training samples.';
for(let k=0;k<2;k++)$('#rs-'+(k?'res':'plain')+'-caption').textContent=view.pca?view.dim+' features → 2 PCs · '+(view.retained[k]===null?'no within-network variance':(100*view.retained[k]).toFixed(1)+'% of this network’s variance retained'):view.dim+' hidden feature'+(view.dim===1?' (shown on a line).':'s, shown directly.');
const table=$('#rs-values');table.replaceChildren();const row=(name,a,b)=>{const tr=document.createElement('tr');for(const v of [name,a,b]){const td=document.createElement('td');td.textContent=v;tr.append(td)}table.append(tr)},fmt=a=>'['+Array.from(a).map(v=>v.toFixed(4)).join(', ')+']';
row('Input',fmt([x,y]),fmt([x,y]));for(let l=0;l<=depth;l++){row((l===0?'After stem':'After block '+l)+' · '+widths[l]+'D',fmt(selected[0].hs[l+1]),fmt(selected[1].hs[l+1]));if(l===layer)table.lastChild.classList.add('rs-selected-row')}row('Scalar prediction',selected[0].pred.toFixed(4),selected[1].pred.toFixed(4));row('Target',target(x,y).toFixed(4),target(x,y).toFixed(4));
if(mseCache?.owner!==nets||mseCache.step!==step)mseCache={owner:nets,step,values:nets.map(n=>n.mse(points))};const mse=mseCache.values;
$('#rs-loss').textContent='MSE on all '+points.length.toLocaleString()+' training examples: plain '+mse[0].toFixed(4)+' · residual '+mse[1].toFixed(4)+'. '+(samplesSeen/points.length).toFixed(2)+' dataset epochs · '+inspection.singleSteps+' single-example steps · identical shuffled batches of '+batchSize+' · Adam 0.003. Only the final scalar prediction is supervised.';drawSurfaces();drawInspector()}
function update(){if(batchCursor>=order.length){shuffle();batchCursor=0}const batch=order.slice(batchCursor,batchCursor+batchSize).map(i=>points[i]);for(const n of nets)n.update(batch,.003,'Adam');inspection.lastBatch=batch;inspection.lastKind='batch';batchCursor+=batch.length;samplesSeen+=batch.length;step++;recordInspection('batch');$('#rs-example-result').textContent=''}
/*__INSPECTOR__*/
function setAllWidths(value){const w=Math.max(1,Math.min(32,Math.round(Number(value)||1)));widths.fill(w);$('#rs-width-all').value=w;$('#rs-width-all-number').value=w;for(let i=0;i<widths.length;i++){$('#rs-width-'+i).value=w;$('#rs-width-number-'+i).value=w}init()}
$('#rs-width-all').oninput=()=>setAllWidths($('#rs-width-all').value);$('#rs-width-all-number').onchange=()=>setAllWidths($('#rs-width-all-number').value);
$('#rs-apply-width').onclick=()=>setAllWidths($('#rs-width-all-number').value);
function setSamples(value){const n=Math.max(128,Math.min(4096,Math.round(Number(value)/128)*128||128));$('#rs-samples').value=n;$('#rs-samples-number').value=n;init()}
$('#rs-samples').oninput=()=>setSamples($('#rs-samples').value);$('#rs-samples-number').onchange=()=>setSamples($('#rs-samples-number').value);
$('#rs-init').onclick=init;$('#rs-depth').onchange=()=>{const count=Number($('#rs-depth').value)+1;while(widths.length<count)widths.push(Number($('#rs-width-all').value));widths=widths.slice(0,count);buildWidthControls();init()};$('#rs-seed').onchange=init;$('#rs-layer').oninput=draw;
$('#rs-layer-number').onchange=()=>{const input=$('#rs-layer-number'),range=$('#rs-layer'),v=Number(input.value);if(input.value!==''&&Number.isFinite(v))range.value=Math.max(Number(range.min),Math.min(Number(range.max),v));draw()};
const inputGrid=$('#rs-input');let inputDrag=null,inputFrame=null;
function setInputPoint(x,y,defer=false){
 if(!Number.isFinite(x)||!Number.isFinite(y))return;
 const clamp=v=>Math.round(Math.max(-2,Math.min(2,v))*100)/100;
 inputPoint.x=clamp(x);inputPoint.y=clamp(y);
 if(defer){if(inputFrame===null)inputFrame=requestAnimationFrame(()=>{inputFrame=null;draw()})}
 else{if(inputFrame!==null)cancelAnimationFrame(inputFrame);inputFrame=null;draw()}
}
function gridPoint(event){
 const matrix=inputGrid.getScreenCTM();if(!matrix)return null;
 // Convert through the SVG transform, including responsive scaling and scrolling.
 const point=new DOMPoint(event.clientX,event.clientY).matrixTransform(matrix.inverse());
 return{x:(point.x-125)/45,y:(111-point.y)/45};
}
function dragInput(event,defer){
 const point=gridPoint(event);if(point)setInputPoint(point.x+inputDrag.dx,point.y+inputDrag.dy,defer);
}
inputGrid.onpointerdown=event=>{
 if(!event.isPrimary||event.button!==0||inputDrag)return;
 const point=gridPoint(event);if(!point)return;
 const handle=event.target.closest('[data-input-handle],.rs-selected-point');
 if(!handle&&(Math.abs(point.x)>2||Math.abs(point.y)>2))return;
 event.preventDefault();inputGrid.focus({preventScroll:true});
 inputDrag={id:event.pointerId,dx:handle?inputPoint.x-point.x:0,dy:handle?inputPoint.y-point.y:0};
 inputGrid.setPointerCapture(event.pointerId);inputGrid.classList.add('is-dragging');dragInput(event,false);
};
inputGrid.onpointermove=event=>{if(inputDrag?.id===event.pointerId){event.preventDefault();dragInput(event,true)}};
function endInputDrag(event,commit){
 if(inputDrag?.id!==event.pointerId)return;
 if(commit)dragInput(event,false);
 else if(inputFrame!==null){cancelAnimationFrame(inputFrame);inputFrame=null;draw()}
 inputDrag=null;inputGrid.classList.remove('is-dragging');
 if(inputGrid.hasPointerCapture(event.pointerId))inputGrid.releasePointerCapture(event.pointerId);
}
inputGrid.onpointerup=event=>endInputDrag(event,true);
inputGrid.onpointercancel=inputGrid.onlostpointercapture=event=>endInputDrag(event,false);
inputGrid.onkeydown=event=>{
 const directions={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,1],ArrowDown:[0,-1]},direction=directions[event.key];
 if(event.key==='Home'){event.preventDefault();setInputPoint(0,0)}
 else if(direction){event.preventDefault();const distance=event.shiftKey ? 0.01 : 0.05;setInputPoint(inputPoint.x+direction[0]*distance,inputPoint.y+direction[1]*distance)}
};
$('#rs-view-left').onclick=()=>{camera.yaw-=Math.PI/12;drawSurfaces()};$('#rs-view-right').onclick=()=>{camera.yaw+=Math.PI/12;drawSurfaces()};
for(const canvas of root.querySelectorAll('.rs-surface')){
 let drag=null;canvas.onpointerdown=e=>{if(e.button!==0)return;drag={x:e.clientX,y:e.clientY,yaw:camera.yaw,pitch:camera.pitch};canvas.setPointerCapture(e.pointerId)};
 canvas.onpointermove=e=>{if(!drag)return;camera.yaw=drag.yaw+(e.clientX-drag.x)*.012;camera.pitch=Math.max(.18,Math.min(1.15,drag.pitch-(e.clientY-drag.y)*.006));drawSurfaces()};
 canvas.onpointerup=canvas.onpointercancel=()=>{drag=null};
}
function runUpdates(count){
 if(running)return;
 runStart=step;runGoal=count;running=true;updateTrainingUI();
 const tick=()=>{
  if(!running)return;
  try{
   update();
   if(!mseCache.values.every(Number.isFinite))throw Error('Non-finite loss');
   if(step-runStart>=runGoal)stop('Completed '+runGoal.toLocaleString()+' additional updates. Total '+step.toLocaleString()+'.');
   updateTrainingUI();draw();
   if(running)trainTimer=setTimeout(tick,matchMedia('(prefers-reduced-motion: reduce)').matches?80:10);
  }catch(error){stop('Training stopped: '+error.message+'. Reset weights to start again.');}
 };
 trainTimer=setTimeout(tick,0);
}
$('#rs-step').onclick=()=>{stop();update();updateTrainingUI();$('#rs-run-status').textContent='Completed one update. Total '+step+'.';draw()};
$('#rs-run').onclick=()=>runUpdates(100);
$('#rs-run-500').onclick=()=>runUpdates(500);
$('#rs-run-1000').onclick=()=>runUpdates(1000);
$('#rs-pause').onclick=()=>{stop('Paused at update '+step+'. Choose a training button to continue.');draw()};
function loadPreset(depth,width){
 $('#rs-depth').value=depth;$('#rs-width-all').value=$('#rs-width-all-number').value=width;
 widths=Array(depth+1).fill(width);$('#rs-layer').value=Math.min(depth,Number($('#rs-layer').max));buildWidthControls();init();$('#rs-layer').value=depth;draw();
}
$('#preset-deep').onclick=()=>loadPreset(36,10);
$('#preset-small').onclick=()=>loadPreset(3,2);
function applyVisibility(){
 const heat=$('#show-heatmaps').checked,slice=$('#show-slices').checked;
 for(const [checkbox,panel] of [['show-architecture','rs-configuration-panel'],['show-widths','rs-width-details'],['show-correction','rs-correction-panel'],['show-values','rs-values-panel'],['show-heatmaps','rs-heatmap-panel'],['show-slices','rs-slice-panel']])$('#'+panel).hidden=!$('#'+checkbox).checked;
 $('#rs-inspector').hidden=!(heat||slice);
 root.classList.toggle('with-correction',$('#show-correction').checked);
 if(nets)draw();
}
for(const input of root.querySelectorAll('.workspace-options input,#show-widths'))input.addEventListener('change',applyVisibility);
const compactWorkspace=matchMedia('(max-width:1099px)');
function setWorkspaceOpen(open,restoreFocus=false){
 const visible=compactWorkspace.matches&&open;
 root.classList.toggle('workspace-open',visible);
 $('#workspace-toggle').setAttribute('aria-expanded',String(visible));
 $('.workspace-backdrop').hidden=!visible;
 if(visible)$('#show-architecture').focus({preventScroll:true});
 else if(restoreFocus&&compactWorkspace.matches)$('#workspace-toggle').focus({preventScroll:true});
}
$('#workspace-toggle').onclick=()=>setWorkspaceOpen(!root.classList.contains('workspace-open'),true);
$('.workspace-backdrop').onclick=()=>setWorkspaceOpen(false,true);
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&root.classList.contains('workspace-open')){event.preventDefault();setWorkspaceOpen(false,true)}});
compactWorkspace.addEventListener('change',()=>setWorkspaceOpen(false));
// Measure actual wrapping, including status text, so the fixed bar never covers
// the start of the page or an anchor target after a viewport/font-size change.
function syncTrainingDock(){
 const height=Math.ceil($('.training-toolbar').getBoundingClientRect().height)+'px';
 if(document.documentElement.style.getPropertyValue('--training-dock-height')!==height)document.documentElement.style.setProperty('--training-dock-height',height);
}
new ResizeObserver(syncTrainingDock).observe($('.training-toolbar'));syncTrainingDock();
root.__test={runUpdates,applyVisibility,loadPreset,Net,target,fitPCA,projectPCA,update,draw,surfaceData,featureData,gradientData,sliceData,lineChart,trainSelectedExample,getState:()=>({nets,step,points,camera,inputPoint:{...inputPoint},widths:widths.slice(),samplesSeen,batchCursor,order:order.slice(),running,history:inspection.history,lastBatch:inspection.lastBatch,singleSteps:inspection.singleSteps})};buildWidthControls();init();
applyVisibility();
new ResizeObserver(()=>{drawSurfaces();drawInspector()}).observe($('#rs-target-surface'));
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>drawSurfaces());
})();
