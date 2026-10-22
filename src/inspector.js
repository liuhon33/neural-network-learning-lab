// Embedded inside the lab closure by scripts/build.py.
// Evaluation always uses current parameters. The last batch is retained as data,
// not as a stale pre-update gradient. Coordinate sweeps never train the model.
const inspection={history:[],lastBatch:null,lastKind:null,singleSteps:0,gradCache:null,sliceCache:null};
const modelNames=['Plain','Residual'],modelColors=['var(--viz-series-1)','var(--viz-series-2)'];
const shortNumber=v=>!Number.isFinite(v)?'not finite':v===0?'0':Math.abs(v)<.001||Math.abs(v)>=1e4?v.toExponential(2):Number(v.toPrecision(5)).toString();
function selectedExample(){const {x,y}=inputPoint;return[x,y,target(x,y)]}
function layerName(l){return l===0?'stem':l===nets[0].depth+1?'head':'block '+l}
function inspectorReset(){
 inspection.history=[];inspection.lastBatch=null;inspection.lastKind=null;inspection.singleSteps=0;inspection.gradCache=inspection.sliceCache=null;
 $('#rs-inspect-layer').max=$('#rs-inspect-layer-number').max=nets[0].depth+1;
 $('#rs-inspect-layer').value=Math.min(Number($('#rs-inspect-layer').value),nets[0].depth+1);
 $('#rs-preview-offset').value=$('#rs-preview-number').value=0;$('#rs-example-result').textContent='';
 recordInspection('initial');
}
function recordInspection(kind){
 if(inspection.history.at(-1)?.step===step)return;
 if(mseCache?.owner!==nets||mseCache.step!==step)mseCache={owner:nets,step,values:nets.map(n=>n.mse(points))};
 const data=surfaceData(),test=data.values.slice(1).map(values=>values.reduce((sum,p,i)=>sum+(p-data.values[0][i])**2,0)/values.length);
 inspection.history.push({step,kind,train:mseCache.values.slice(),test});
 inspection.gradCache=inspection.sliceCache=null;$('#rs-preview-offset').value=$('#rs-preview-number').value=0;
}
function gradientData(){
 const scope=$('#rs-gradient-source').value,point=selectedExample(),key=[step,scope,...(scope==='single'?point:[])].join('|');
 if(inspection.gradCache?.owner===nets&&inspection.gradCache.key===key)return inspection.gradCache;
 const batch=scope==='full'?points:scope==='single'?[point]:inspection.lastBatch||order.slice(0,batchSize).map(i=>points[i]);
 const label=scope==='full'?'Full training set ('+batch.length+' examples)':scope==='single'?'Selected example ('+point.slice(0,2).join(', ')+')':inspection.lastBatch?'Latest update’s '+(inspection.lastKind==='single'?'single example':'batch ('+batch.length+' examples)'):'First shuffled batch ('+batch.length+' examples; no update yet)';
 inspection.gradCache={owner:nets,key,scope,batch,label,values:nets.map(n=>n.gradients(batch))};return inspection.gradCache;
}
function selection(){
 const layer=Math.max(0,Math.min(nets[0].depth+1,Number($('#rs-inspect-layer').value)||0)),key=$('#rs-tensor').value,z=nets[0].layers[layer];
 const rows=z.no,cols=key==='b'?1:z.ni,row=Math.max(0,Math.min(rows-1,Math.floor(Number($('#rs-weight-row').value)||0))),col=Math.max(0,Math.min(cols-1,Math.floor(Number($('#rs-weight-col').value)||0)));
 $('#rs-inspect-layer-number').value=layer;$('#rs-inspect-layer-label').textContent=layerName(layer);
 $('#rs-weight-row').max=rows-1;$('#rs-weight-col').max=cols-1;$('#rs-weight-row').value=row;$('#rs-weight-col').value=col;$('#rs-weight-col').disabled=cols===1;
 const symbol=key==='b'?'b':key==='skip'?'P':'W';
 return{layer,key,rows,cols,row,col,index:row*cols+col,name:symbol+'['+row+(key==='b'?'':','+col)+']'};
}
function parameterArray(net,s){const z=net.layers[s.layer];return net.keys(z).includes(s.key)?z[s.key]:null}
function absentParameter(k,s){return k===0?'No shortcut parameters':s.layer>0&&s.layer<=nets[1].depth?'Identity shortcut · no trainable P':'No shortcut at '+layerName(s.layer)}
function sliderPaint(){for(const el of $('#rs-inspector').querySelectorAll('input[type=range]')){const span=Number(el.max)-Number(el.min);el.style.setProperty('--rs-fill',(span?100*(Number(el.value)-Number(el.min))/span:0)+'%')}}
function emptyChart(svg,message){const width=Math.max(300,Math.min(560,svg.getBoundingClientRect().width||560));svg.replaceChildren();svg.setAttribute('viewBox','0 0 '+width+' 300');svg.append(make('text',{x:width/2,y:145,'text-anchor':'middle'},message))}
function lineChart(svg,series,{xlabel,ylabel,log=false,xmin=null,xmax=null,ymin=null,ymax=null,ticks=null}={}){
 const width=Math.max(300,Math.min(560,svg.getBoundingClientRect().width||560)),right=width-25;
 svg.replaceChildren();svg.setAttribute('viewBox','0 0 '+width+' 300');
 const finite=series.flatMap(s=>s.data).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));
 if(!finite.length){emptyChart(svg,'No finite values');return null}
 const smallestPositive=finite.reduce((a,p)=>p[1]>0?Math.min(a,p[1]):a,Infinity);
 const zeroFloor=Number.isFinite(smallestPositive)?Math.max(Number.MIN_VALUE,Math.min(1e-16,smallestPositive/10)):1e-16;
 // Preserve genuinely tiny positive gradients; only exact zero needs a floor.
 const transform=v=>log?Math.log10(v>0?v:zeroFloor):v;
 svg.dataset.zeroFloor=zeroFloor;
 const xx=finite.map(p=>p[0]),yy=finite.map(p=>transform(p[1]));
 const extent=values=>values.reduce((a,v)=>[Math.min(a[0],v),Math.max(a[1],v)],[Infinity,-Infinity]);
 const xr=extent(xx),yr=extent(yy);xmin=xmin??xr[0];xmax=xmax??xr[1];
 if(xmin===xmax)xmax=xmin+1;
 ymin=ymin??(log?Math.floor(yr[0]):Math.min(0,yr[0]));ymax=ymax??(log?Math.ceil(yr[1]):yr[1]*1.08);
 if(ymax<=ymin)ymax=ymin+1;
 const X=x=>70+(x-xmin)/(xmax-xmin)*(right-70),Y=y=>250-(transform(y)-ymin)/(ymax-ymin)*210;
 for(let i=0;i<=4;i++){
  const y=ymin+(ymax-ymin)*i/4,py=250-i*210/4;
  svg.append(make('path',{d:'M70,'+py+'H'+right,stroke:'var(--border)',fill:'none','data-grid-tick':y}));
  svg.append(make('text',{x:62,y:py+4,'text-anchor':'end','data-y-tick':y},log?'10^'+Number(y.toFixed(1)):shortNumber(Number(y.toPrecision(3)))));
 }
 const divisions=width<400?2:4;
 const xt=ticks||Array.from({length:divisions+1},(_,i)=>{const x=xmin+(xmax-xmin)*i/divisions;return[x,shortNumber(Number(x.toPrecision(3)))]});
 for(const[x,label]of xt)svg.append(make('text',{x:X(x),y:270,'text-anchor':x===xmin?'start':x===xmax?'end':'middle'},label));
 svg.append(make('text',{x:70,y:22},ylabel),make('text',{x:(70+right)/2,y:295,'text-anchor':'middle'},xlabel));
 for(const s of series){
  let path='',active=false;
  for(const p of s.data){if(!Number.isFinite(p[1])){active=false;continue}path+=(active?'L':'M')+X(p[0])+','+Y(p[1]);active=true}
  if(!s.pointsOnly)svg.append(make('path',{d:path,stroke:s.color,'stroke-width':1.8,'stroke-dasharray':s.dash||'',fill:'none'}));
  const marks=s.noMarks?[]:s.markAll?s.data:s.data.length?[s.data.at(-1)]:[];
  for(const p of marks){if(!Number.isFinite(p[1]))continue;const attrs=s.square?{x:X(p[0])-3,y:Y(p[1])-3,width:6,height:6}:{cx:X(p[0]),cy:Y(p[1]),r:3};const dot=make(s.square?'rect':'circle',{...attrs,fill:s.color});dot.append(make('title',{},s.name+': '+shortNumber(p[1])));svg.append(dot)}
 }
 return{X,Y,xmin,xmax,ymin,ymax};
}
function drawHeatmap(svg,values,s,maxValue,k,isGradient){
 svg.replaceChildren();svg.setAttribute('viewBox','0 0 360 358');svg.dataset.model=k;svg.dataset.gradient=Number(isGradient);svg.dataset.limit=maxValue;
 if(!values){svg.append(make('text',{x:180,y:148,'text-anchor':'middle'},absentParameter(k,s)));return}
 const cell=Math.min(250/s.cols,220/s.rows),left=58+(250-cell*s.cols)/2,top=26+(220-cell*s.rows)/2;
 svg.append(make('rect',{x:left,y:top,width:s.cols*cell,height:s.rows*cell,fill:'var(--card)'}));
 for(let row=0;row<s.rows;row++)for(let col=0;col<s.cols;col++){
  const index=row*s.cols+col,v=values[index],alpha=maxValue>0?Math.min(1,Math.abs(v)/maxValue):0;
  const rect=make('rect',{x:left+col*cell,y:top+row*cell,width:cell,height:cell,fill:v<0?'var(--viz-series-4)':'var(--viz-series-5)','fill-opacity':alpha,'data-index':index,'data-row':row,'data-col':col,'data-value':v});
  svg.append(rect);
 }
 svg.append(make('rect',{x:left+s.col*cell,y:top+s.row*cell,width:cell,height:cell,fill:'none',stroke:'var(--foreground)','stroke-width':2,'pointer-events':'none'}));
 for(const col of [...new Set([0,s.cols-1])])svg.append(make('text',{x:left+(col+.5)*cell,y:264,'text-anchor':'middle'},col));
 for(const row of [...new Set([0,s.rows-1])])svg.append(make('text',{x:46,y:top+(row+.5)*cell+4,'text-anchor':'end'},row));
 svg.append(make('text',{x:183,y:283,'text-anchor':'middle'},s.key==='b'?'bias column':'input column'),make('text',{x:8,y:15},'output row'),make('text',{x:183,y:306,'text-anchor':'middle'},'color bounds ±'+shortNumber(maxValue)));

 const gradientId=svg.id+'-scale',defs=make('defs',{}),gradient=make('linearGradient',{id:gradientId,x1:'0%',x2:'100%'});
 for(const [offset,color]of [['0%','var(--viz-series-4)'],['50%','var(--card)'],['100%','var(--viz-series-5)']])gradient.append(make('stop',{offset,'stop-color':maxValue===0?'var(--card)':color}));
 defs.append(gradient);svg.append(defs);
 svg.append(make('rect',{x:58,y:317,width:250,height:12,fill:'url(#'+gradientId+')',stroke:'var(--border)','data-colorbar':'true'}));
 for(const [x,value,anchor]of [[58,-maxValue,'start'],[183,0,'middle'],[308,maxValue,'end']])svg.append(make('text',{x,y:348,'text-anchor':anchor,'data-color-tick':value},shortNumber(value)));
}
function sliceData(){
 const s=selection(),point=selectedExample(),radius=Number($('#rs-sweep-radius').value),key=[step,s.layer,s.key,s.index,...point,radius].join('|');
 if(inspection.sliceCache?.owner===nets&&inspection.sliceCache.key===key)return inspection.sliceCache;
 const centers=nets.map(n=>parameterArray(n,s)?.[s.index]??null),valid=centers.filter(v=>v!==null),xmin=valid.length?Math.min(...valid)-radius:-radius,xmax=valid.length?Math.max(...valid)+radius:radius;
 const values=Array.from({length:101},(_,i)=>xmin+(xmax-xmin)*i/100);
 const curves=nets.map((n,k)=>centers[k]===null?null:{center:centers[k],points:n.weightSlice(point,s.layer,s.key,s.index,values),currentLoss:n.mse([point]),gradient:n.gradients([point]).grads[s.layer][s.key][s.index]});
 inspection.sliceCache={owner:nets,key,s,point,radius,xmin,xmax,curves};return inspection.sliceCache;
}
function drawSlices(){
 if($('#rs-slice-panel').hidden)return;
 const s=selection(),point=selectedExample(),radius=Number($('#rs-sweep-radius').value);
 $('#rs-sweep-radius-value').textContent=radius.toFixed(2);
 for(const el of [$('#rs-preview-offset'),$('#rs-preview-number')]){el.min=-radius;el.max=radius;el.value=Math.max(-radius,Math.min(radius,Number(el.value)||0))}
 const offset=Number($('#rs-preview-offset').value);$('#rs-preview-number').value=offset;
 for(const id of ['rs-preview-offset','rs-preview-number','rs-sweep-radius','rs-example-step'])$('#'+id).disabled=running;
 $('#rs-example-context').textContent='Input ('+point.slice(0,2).map(v=>v.toFixed(2)).join(', ')+') · target '+shortNumber(point[2])+' · '+layerName(s.layer)+' '+s.name+'. '+(running?'Pause training to explore a fixed model.':'The same preview change Δw is applied separately to each network’s current parameter.');
 if(running){for(const id of ['plain','res']){emptyChart($('#rs-'+id+'-slice'),'Pause training to inspect this loss curve');$('#rs-'+id+'-slice-summary').textContent=''}sliderPaint();return}
 const data=sliceData(),finiteLoss=data.curves.filter(Boolean).flatMap(c=>c.points.map(p=>p.loss)).filter(Number.isFinite),ymax=Math.max(1e-8,finiteLoss.reduce((a,v)=>Math.max(a,v),0)*1.08);
 data.curves.forEach((c,k)=>{
  const prefix=k?'res':'plain',svg=$('#rs-'+prefix+'-slice'),summary=$('#rs-'+prefix+'-slice-summary');
  if(!c){emptyChart(svg,absentParameter(k,s));summary.textContent='No trainable parameter to sweep here.';for(const key of ['currentWeight','currentLoss','gradient','previewWeight','previewLoss'])delete svg.dataset[key];return}
  const axes=lineChart(svg,[{name:modelNames[k],color:modelColors[k],noMarks:true,data:c.points.map(p=>[p.value,p.loss])}],{xlabel:s.name+' (actual parameter value)',ylabel:'single-example squared error',xmin:data.xmin,xmax:data.xmax,ymin:0,ymax});
  if(!axes)return;
  const preview=nets[k].weightSlice(point,s.layer,s.key,s.index,[c.center+offset])[0],{X,Y}=axes;
  // Keep the tangent inside the chart, independently of the sweep's y range.
  const half=Math.min(radius/5,Math.abs(c.gradient)>1e-15?ymax/(6*Math.abs(c.gradient)):radius/5);
  let dmin=-half,dmax=half;
  if(Math.abs(c.gradient)>1e-15){const limits=[-c.currentLoss/c.gradient,(ymax-c.currentLoss)/c.gradient].sort((a,b)=>a-b);dmin=Math.max(dmin,limits[0]);dmax=Math.min(dmax,limits[1])}
  const tangent=[dmin,dmax].map(d=>[c.center+d,c.currentLoss+c.gradient*d]);
  svg.append(make('path',{d:tangent.map((p,i)=>(i?'L':'M')+X(p[0])+','+Y(p[1])).join(''),stroke:'var(--foreground)','stroke-dasharray':'4 3',fill:'none'}));
  svg.append(make('circle',{cx:X(c.center),cy:Y(c.currentLoss),r:5,fill:modelColors[k],stroke:'var(--foreground)'}));
  const px=X(preview.value),py=Y(preview.loss);svg.append(make('path',{d:'M'+px+','+(py-6)+'l6,6 -6,6 -6,-6Z',fill:'var(--background)',stroke:modelColors[k],'stroke-width':2}));
  Object.assign(svg.dataset,{currentWeight:c.center,currentLoss:c.currentLoss,gradient:c.gradient,previewWeight:preview.value,previewLoss:preview.loss});
  summary.textContent='Current '+s.name+' = '+shortNumber(c.center)+' · loss '+shortNumber(c.currentLoss)+' · ∂loss/∂parameter '+shortNumber(c.gradient)+'. Preview '+shortNumber(preview.value)+' → loss '+shortNumber(preview.loss)+'.';
 });sliderPaint();
}
function drawLearningCharts(g){
 const history=inspection.history,series=[];
 for(let k=0;k<2;k++)for(const kind of ['train','test'])series.push({name:modelNames[k]+' '+kind,color:modelColors[k],square:k===1,dash:kind==='test'?'5 4':'',data:history.map(h=>[h.step,h[kind][k]])});
 lineChart($('#rs-loss-chart'),series,{xlabel:'optimizer update',ylabel:'MSE · log scale',log:true,xmin:0,xmax:Math.max(1,step)});
 const last=history.at(-1);
 $('#rs-history-summary').textContent='Plain: blue, circles · residual: orange, squares. Solid: all '+points.length+' training examples. Dashed: fixed '+surfaceGrid.length+'-point evaluation grid. Current grid MSE '+last.test.map(shortNumber).join(' / ')+'. '+inspection.singleSteps+' single-example update(s) included. Exact zeros use an adaptive floor on log axes.';
 const rms=$('#rs-gradient-measure').value==='rms',norm=arr=>Math.hypot(...arr)/(rms?Math.sqrt(arr.length):1);
 const gs=g.values.map((v,k)=>({name:modelNames[k]+' main weights',color:modelColors[k],square:k===1,markAll:true,data:v.grads.map((z,l)=>[l,norm(z.w)])}));
 const projected=g.values[1].grads.flatMap((z,l)=>z.skip?[[l,norm(z.skip)]]:[]);
 if(projected.length)gs.push({name:'Residual shortcut weights',color:modelColors[1],pointsOnly:true,markAll:true,data:projected});
 const lastLayer=nets[0].depth+1,spacing=Math.max(1,Math.ceil(lastLayer/6)),ticks=[[0,'stem']];
 for(let l=spacing;l<lastLayer;l+=spacing)if(lastLayer-l>spacing*.65)ticks.push([l,String(l)]);ticks.push([lastLayer,'head']);
 const axes=lineChart($('#rs-gradient-chart'),gs,{xlabel:'layer · stem → blocks → head',ylabel:rms?'RMS gradient per weight':'gradient norm',log:true,xmin:0,xmax:lastLayer,ticks});
 const s=selection();
 if(axes)for(const v of gs.slice(0,2)){const p=v.data[s.layer];$('#rs-gradient-chart').append(make('circle',{cx:axes.X(p[0]),cy:axes.Y(p[1]),r:7,fill:'none',stroke:v.color,'stroke-width':2}))}
 $('#rs-norm-summary').textContent=g.label+'; recomputed at current weights. Main W matrices only; biases excluded.'+(projected.length?' Extra orange circles: learned shortcut P gradients.':'')+' '+layerName(s.layer)+': plain '+shortNumber(gs[0].data[s.layer][1])+' / residual '+shortNumber(gs[1].data[s.layer][1])+'. '+(rms?'RMS divides the sum of squared gradients by the number of weights before taking the square root.':'Norm grows with both gradient size and the number of weights.');
}
function drawInspector(){
 const s=selection(),g=gradientData(),weights=nets.map(n=>parameterArray(n,s)),grads=g.values.map((v,k)=>weights[k]?v.grads[s.layer][s.key]:null);
 $('#rs-heatmap-hover').textContent='Point to a cell for its exact value.';
 const maximum=arrays=>arrays.filter(Boolean).reduce((m,a)=>a.reduce((v,x)=>Math.max(v,Math.abs(x)),m),0),wmax=maximum(weights),gmax=maximum(grads);
 $('#rs-gradient-context').textContent=g.label+' · gradients evaluated at the currently displayed weights (update '+step+'). These are raw derivatives, before Adam rescales them. Changing the example selector does not train either model.';
 if(!$('#rs-heatmap-panel').hidden)for(let k=0;k<2;k++){const prefix=k?'res':'plain';drawHeatmap($('#rs-'+prefix+'-weights'),weights[k],s,wmax,k,false);drawHeatmap($('#rs-'+prefix+'-gradients'),grads[k],s,gmax,k,true)}
 $('#rs-heatmap-values').textContent=layerName(s.layer)+' · '+s.name+' · '+weights.map((w,k)=>modelNames[k]+': '+(w?'value '+shortNumber(w[s.index])+', gradient '+shortNumber(grads[k][s.index]):'no trainable shortcut')).join(' | ');
 drawSlices();drawLearningCharts(g);
}
function trainSelectedExample(){
 if(running)return;
 const point=selectedExample(),before=nets.map(n=>n.mse([point])),s=selection(),old=nets.map(n=>parameterArray(n,s)?.[s.index]??null);
 for(const n of nets)n.update([point],.003,'Adam');
 inspection.lastBatch=[point];inspection.lastKind='single';inspection.singleSteps++;step++;recordInspection('single');updateTrainingUI();$('#rs-run-status').textContent='Completed one selected-example update. Total '+step+'.';draw();
 $('#rs-example-result').textContent='Update '+step+' on ('+point.slice(0,2).map(v=>v.toFixed(2)).join(', ')+'): '+nets.map((n,k)=>modelNames[k]+' loss '+shortNumber(before[k])+' → '+shortNumber(n.mse([point]))+(old[k]===null?'':' · actual Δ'+s.name+' '+shortNumber(parameterArray(n,s)[s.index]-old[k]))).join(' | ')+'. All parameters were eligible to update; these are newly evaluated curves.';
}
$('#rs-inspect-layer').oninput=()=>{inspection.sliceCache=null;$('#rs-preview-offset').value=0;drawInspector()};
$('#rs-inspect-layer-number').onchange=()=>{const el=$('#rs-inspect-layer');el.value=Math.max(0,Math.min(Number(el.max),Math.floor(Number($('#rs-inspect-layer-number').value)||0)));el.oninput()};
for(const id of ['rs-tensor','rs-weight-row','rs-weight-col'])$('#'+id).onchange=()=>{inspection.sliceCache=null;$('#rs-preview-offset').value=0;drawInspector()};
for(const id of ['rs-gradient-source','rs-gradient-measure'])$('#'+id).onchange=drawInspector;
$('#rs-sweep-radius').oninput=drawSlices;$('#rs-preview-offset').oninput=drawSlices;
$('#rs-preview-number').onchange=()=>{const el=$('#rs-preview-offset');el.value=Math.max(Number(el.min),Math.min(Number(el.max),Number($('#rs-preview-number').value)||0));drawSlices()};
$('#rs-example-step').onclick=trainSelectedExample;
for(const svg of root.querySelectorAll('.rs-heatmap')){
 svg.onclick=e=>{const cell=e.target.closest('[data-index]');if(!cell)return;$('#rs-weight-row').value=cell.dataset.row;$('#rs-weight-col').value=cell.dataset.col;$('#rs-preview-offset').value=0;inspection.sliceCache=null;drawInspector()};
 svg.onpointermove=e=>{const cell=e.target.closest('[data-index]');if(!cell)return;$('#rs-heatmap-hover').textContent=modelNames[Number(svg.dataset.model)]+' · row '+cell.dataset.row+', column '+cell.dataset.col+' · '+(svg.dataset.gradient==='1'?'gradient ':'parameter ')+Number(cell.dataset.value).toPrecision(10)};
}
