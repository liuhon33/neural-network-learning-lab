// Numerical model and PCA, embedded in the standalone lab.
function rng(seed){let a=seed|0;return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gaussian(r){return Math.sqrt(-2*Math.log(Math.max(1e-12,r())))*Math.cos(2*Math.PI*r())}
function target(x,y){return 1.3*Math.exp(-2.3*((x+.8)**2+(y-.5)**2))+.9*Math.exp(-3*((x-.9)**2+(y+.6)**2))-1.05*Math.exp(-2.8*((x-.45)**2+(y-.9)**2))+.18*x*y}
function trainingPoints(count){const r=rng(1234);return Array.from({length:count},()=>{const x=r()*4-2,y=r()*4-2;return[x,y,target(x,y)]})}
class Net{
 constructor(widths,res,seed,width=16){
  this.widths=Array.isArray(widths)?widths.slice():Array(widths+1).fill(width);
  if(this.widths.length<2||this.widths.some(w=>!Number.isInteger(w)||w<1))throw Error('Invalid hidden widths');
  this.depth=this.widths.length-1;this.res=res;this.alpha=1/Math.sqrt(this.depth);this.t=0;this.layers=[];
  const r=rng(seed),dims=[2,...this.widths,1];
  for(let l=0;l<dims.length-1;l++){
   const ni=dims[l],no=dims[l+1],z={ni,no,w:Float64Array.from({length:ni*no},()=>gaussian(r)*Math.sqrt(2/(ni+no))),b:new Float64Array(no)};
   // A width change uses a trainable, bias-free rectangular identity shortcut.
   // No extra random draws: corresponding main-branch weights remain identical.
   if(res&&l>0&&l<=this.depth&&ni!==no){z.skip=new Float64Array(ni*no);for(let i=0;i<Math.min(ni,no);i++)z.skip[i*ni+i]=1}
   for(const key of this.keys(z)){z['m'+key]=new Float64Array(z[key].length);z['v'+key]=new Float64Array(z[key].length)}
   this.layers.push(z);
  }
 }
 keys(z){return z.skip?['w','b','skip']:['w','b']}
 parameterCount(){return this.layers.reduce((s,z)=>s+this.keys(z).reduce((q,k)=>q+z[k].length,0),0)}
 forward(x,y,stopAt=this.depth){
  let h=[x,y];const hs=[h],acts=[];
  for(let l=0;l<=stopAt;l++){
   const z=this.layers[l],a=new Float64Array(z.no),out=new Float64Array(z.no);
   for(let j=0;j<z.no;j++){
    let s=z.b[j];for(let i=0;i<z.ni;i++)s+=z.w[j*z.ni+i]*h[i];a[j]=Math.tanh(s);
    if(this.res&&l>0){let bypass=z.skip?0:h[j];if(z.skip)for(let i=0;i<z.ni;i++)bypass+=z.skip[j*z.ni+i]*h[i];out[j]=bypass+this.alpha*a[j]}else out[j]=a[j];
   }
   acts.push(a);h=out;hs.push(h);
  }
  let pred;if(stopAt===this.depth){const z=this.layers.at(-1);pred=z.b[0];for(let i=0;i<z.ni;i++)pred+=z.w[i]*h[i]}
  return{pred,hs,acts};
 }
 gradients(batch){
  const grads=this.layers.map(z=>Object.fromEntries(this.keys(z).map(k=>[k,new Float64Array(z[k].length)])));let loss=0;
  for(const[x,y,t]of batch){
   const c=this.forward(x,y),error=c.pred-t,d=2*error/batch.length,head=this.layers.at(-1),last=grads.at(-1),h=c.hs.at(-1);loss+=error*error;last.b[0]+=d;
   let dh=new Float64Array(head.ni);for(let i=0;i<head.ni;i++){last.w[i]+=d*h[i];dh[i]=head.w[i]*d}
   for(let l=this.depth;l>=0;l--){
    const z=this.layers[l],g=grads[l],a=c.acts[l],prev=c.hs[l],back=new Float64Array(z.ni),residual=this.res&&l>0;
    for(let j=0;j<z.no;j++){
     const dz=dh[j]*(1-a[j]*a[j])*(residual?this.alpha:1);g.b[j]+=dz;
     for(let i=0;i<z.ni;i++){
      g.w[j*z.ni+i]+=dz*prev[i];back[i]+=z.w[j*z.ni+i]*dz;
      if(z.skip){g.skip[j*z.ni+i]+=dh[j]*prev[i];back[i]+=z.skip[j*z.ni+i]*dh[j]}
     }
     if(residual&&!z.skip)back[j]+=dh[j];
    }
    dh=back;
   }
  }
  return{grads,loss:loss/batch.length};
 }
 mse(points){let sum=0;for(const[x,y,t]of points){const e=this.forward(x,y).pred-t;sum+=e*e}return sum/points.length}
 // A coordinate slice is a read-only experiment: restore the parameter even if
 // evaluation fails. Optimizer moments and the update counter are untouched.
 weightSlice(point,layer,key,index,values){
  const z=this.layers[layer],p=z?.[key];
  if(!z||!this.keys(z).includes(key)||!p||!Number.isInteger(index)||index<0||index>=p.length||!values.every(Number.isFinite))throw Error('Invalid parameter slice');
  const original=p[index];
  try{return values.map(value=>{p[index]=value;const prediction=this.forward(point[0],point[1]).pred;return{value,prediction,loss:(prediction-point[2])**2}})}finally{p[index]=original}
 }
 update(batch,lr,opt){
  const g=this.gradients(batch);this.t++;
  this.norms=g.grads.map(q=>Math.sqrt(Object.values(q).reduce((s,v)=>s+v.reduce((a,b)=>a+b*b,0),0)));
  for(let l=0;l<this.layers.length;l++)for(const key of this.keys(this.layers[l])){
   const z=this.layers[l],p=z[key],m=z['m'+key],v=z['v'+key],d=g.grads[l][key];
   for(let i=0;i<p.length;i++)if(opt==='Adam'){m[i]=.9*m[i]+.1*d[i];v[i]=.999*v[i]+.001*d[i]*d[i];p[i]-=lr*(m[i]/(1-.9**this.t))/(Math.sqrt(v[i]/(1-.999**this.t))+1e-8)}else p[i]-=lr*d[i];
  }
 }
}
// Cyclic Jacobi eigensolver for symmetric covariance matrices, up to 32 x 32.
function symmetricEigen(matrix){
 const n=matrix.length,a=matrix.map(r=>Float64Array.from(r)),v=Array.from({length:n},(_,i)=>Float64Array.from({length:n},(_,j)=>Number(i===j)));
 const magnitude=Math.max(1e-30,...a.map((r,i)=>Math.abs(r[i]))),tolerance=magnitude*1e-13;
 for(let sweep=0;sweep<60;sweep++){
  let largest=0;
  for(let p=0;p<n-1;p++)for(let q=p+1;q<n;q++){
   const apq=a[p][q];largest=Math.max(largest,Math.abs(apq));if(Math.abs(apq)<=tolerance)continue;
   const tau=(a[q][q]-a[p][p])/(2*apq),t=(tau>=0?1:-1)/(Math.abs(tau)+Math.hypot(1,tau)),c=1/Math.hypot(1,t),s=t*c;
   a[p][p]-=t*apq;a[q][q]+=t*apq;a[p][q]=a[q][p]=0;
   for(let k=0;k<n;k++){
    if(k!==p&&k!==q){const kp=a[k][p],kq=a[k][q];a[k][p]=a[p][k]=c*kp-s*kq;a[k][q]=a[q][k]=s*kp+c*kq}
    const vp=v[k][p],vq=v[k][q];v[k][p]=c*vp-s*vq;v[k][q]=s*vp+c*vq;
   }
  }
  if(largest<=tolerance)break;
 }
 const order=Array.from({length:n},(_,i)=>i).sort((i,j)=>a[j][j]-a[i][i]);
 return{values:order.map(i=>Math.max(0,a[i][i])),vectors:order.map(i=>Float64Array.from(v,r=>r[i]))};
}
function fitPCA(rows,previous=null){
 const dim=rows[0].length,mean=new Float64Array(dim),cov=Array.from({length:dim},()=>new Float64Array(dim));
 for(const r of rows)for(let i=0;i<dim;i++)mean[i]+=r[i]/rows.length;
 for(const r of rows)for(let i=0;i<dim;i++)for(let j=0;j<=i;j++)cov[i][j]+=(r[i]-mean[i])*(r[j]-mean[j])/rows.length;
 for(let i=0;i<dim;i++)for(let j=0;j<i;j++)cov[j][i]=cov[i][j];
 const eig=symmetricEigen(cov),axes=eig.vectors.slice(0,2);
 for(let k=0;k<axes.length;k++){
  const axis=axes[k],old=previous?.axes[k];let sign;
  if(old?.length===dim)sign=axis.reduce((s,x,i)=>s+x*old[i],0);
  else{let largest=0;for(let i=1;i<dim;i++)if(Math.abs(axis[i])>Math.abs(axis[largest]))largest=i;sign=axis[largest]}
  if(sign<0)for(let i=0;i<dim;i++)axis[i]*=-1;
 }
 const total=eig.values.reduce((a,b)=>a+b,0);
 return{mean,axes,eigenvalues:eig.values,covariance:cov,retained:total>1e-24?(eig.values[0]+(eig.values[1]||0))/total:null};
}
function projectPCA(row,pca){return[0,1].map(k=>pca.axes[k]?pca.axes[k].reduce((s,a,i)=>s+a*(row[i]-pca.mean[i]),0):0)}
function retainedVariance(rows,pca){
 const n=rows.length,d=rows[0].length,mean=new Float64Array(d);for(const r of rows)for(let i=0;i<d;i++)mean[i]+=r[i]/n;
 let total=0,kept=0;for(const r of rows){const centered=r.map((x,i)=>x-mean[i]);total+=centered.reduce((s,x)=>s+x*x,0);for(const a of pca.axes){const dot=a.reduce((s,x,i)=>s+x*centered[i],0);kept+=dot*dot}}
 return total>1e-24?Math.max(0,Math.min(1,kept/total)):null;
}
if(typeof module!=='undefined'&&module.exports)module.exports={Net,rng,target,trainingPoints,symmetricEigen,fitPCA,projectPCA,retainedVariance};
