// Run with: node tests/numerics.cjs. No packages required.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const engine=fs.readFileSync(path.join(__dirname,'../src/dense-engine.js'),'utf8');
const {Net,trainingPoints}=vm.runInNewContext(engine+';({Net,trainingPoints})');
let checked=0,maxError=0;
const batch=trainingPoints(5);
for(const widths of [Array(37).fill(10),[2,2,2,2],[3,2,4]]){
 const models=[new Net(widths,false,7),new Net(widths,true,7)];
 for(let l=0;l<models[0].layers.length;l++)assert.equal(JSON.stringify(models[0].layers[l].w),JSON.stringify(models[1].layers[l].w));
 if(widths.length===37)assert.equal(models[0].parameterCount(),4001),assert.equal(models[1].parameterCount(),4001);
 if(widths.length===4)assert.equal(models[0].parameterCount(),27);
 for(const n of models){
  const gradients=n.gradients(batch).grads;
  for(let l=0;l<n.layers.length;l++)for(const key of n.keys(n.layers[l])){
   const a=n.layers[l][key];
   for(const i of new Set([0,a.length-1])){
    const value=a[i],eps=1e-5;
    a[i]=value+eps;const plus=n.mse(batch);
    a[i]=value-eps;const minus=n.mse(batch);a[i]=value;
    const numeric=(plus-minus)/(2*eps),analytic=gradients[l][key][i];
    const error=Math.abs(numeric-analytic);maxError=Math.max(maxError,error);
    assert.ok(error<2e-7+Math.abs(analytic)*2e-5,`${widths} layer ${l} ${key}[${i}]: ${numeric} vs ${analytic}`);checked++;
   }
  }
  n.update(batch,.003,'Adam');const before=JSON.stringify(n);
  const result=n.weightSlice(batch[0],0,'w',0,[-.2,0,.2]);
  assert.equal(JSON.stringify(n),before,'Sweep changed weights or Adam state');
  assert.ok(result.every(v=>Number.isFinite(v.loss)));
 }
}
console.log(JSON.stringify({status:'passed',checkedDerivatives:checked,maxAbsoluteError:maxError,defaultParameters:4001}));
