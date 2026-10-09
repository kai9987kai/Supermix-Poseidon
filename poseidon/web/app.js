const $=s=>document.querySelector(s);
let mode='chat',history=[],episode=null,frame=0,playing=false,first=true;
let animationHandle=null,atlasReady=false,experimentBusy=false,currentExperiment=null,experimentOutcome='idle';
const plannerNotes={policy:'Direct actions from the trained policy. No teacher fallback.',mpc:'Plans actions using the learned dynamics head. Predicted outcomes can be inaccurate.',hybrid:'Combines the learned policy with dynamics-based planning.',atlas:'Experimental fitted Atlas with explicit decision evidence. Its availability is not evidence of superiority.',risk_aware:'Experimental uncertainty-aware ensemble controller.'};
const controllerNames={policy:'Learned policy',mpc:'Dynamics MPC',neural_mpc:'Neural MPC',hybrid:'Policy + dynamics',atlas:'Counterfactual Atlas',atlas_no_memory:'Atlas without memory',heuristic:'Heuristic control',random:'Random control',risk_aware:'Uncertainty ensemble'};
const controllerName=value=>controllerNames[value]||String(value||'Unspecified controller');
const notes={chat:'Local pretrained language model. Answers can be wrong; arithmetic is checked by a separate exact solver.',math:'Exact rational arithmetic and linear equations. Use explicit multiplication: 3*x + 7 = 22. This result comes from a tool, not unaided neural reasoning.',image:'Learned scene attributes → software renderer. Describe cube, sphere, pyramid or cylinder; color; one to three objects; and small, medium or large.',video:'A learned scene becomes a 2.4-second animation. Motions: still, orbit, bounce, spin. Geometric animation, not photorealistic video.',mesh:'Create basic OBJ + MTL and glTF assets from the same learned scene representation. Up to three geometric objects.',world:'The trained policy chooses six actions in a deterministic synthetic world. Reaches the 256-step horizon or dies. No teacher fallback.'};
const examples={chat:'Ask Poseidon something…',math:'(17 + 5) * 3, or 3*x + 7 = 22',image:'Create two small cyan spheres with orbit motion.',video:'Make three medium purple cubes with bounce motion.',mesh:'Build one large yellow pyramid with still motion.',world:'Run the learned agent in TidePool'};
notes.world='Choose a controller for a deterministic synthetic world. Six actions, a 256-step horizon, and a replay of the actual episode. Atlas is an opt-in experiment.';
$('#planner').onchange=()=>{$('#plannerNote').textContent=plannerNotes[$('#planner').value]||'Experimental survival controller.';};
function setMode(value){mode=value;document.querySelectorAll('[data-mode]').forEach(b=>{b.classList.toggle('selected',b.dataset.mode===mode);b.setAttribute('aria-selected',String(b.dataset.mode===mode));});$('#modeNote').textContent=notes[mode];$('#prompt').placeholder=examples[mode];if($('#plannerControls'))$('#plannerControls').hidden=(mode!=='world');}
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{if(b.dataset.target)setMode(b.dataset.target);$('#prompt').value=b.dataset.prompt;$('#prompt').focus();});
function message(role,text,backend){if(first){$('#messages').replaceChildren();first=false;}let card=document.createElement('div');card.className='message '+role;let label=document.createElement('div');label.className='message-label';label.textContent=role==='user'?'YOU':role==='error'?'REQUEST ERROR':'POSEIDON';let body=document.createElement('div');body.className='message-body';body.textContent=text;card.append(label,body);if(backend){let meta=document.createElement('div');meta.className='message-backend';meta.textContent=backend;card.append(meta);}$('#messages').append(card);$('#messages').scrollTop=$('#messages').scrollHeight;}
async function request(url,payload){const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await response.json();if(!response.ok)throw Error(data.error||response.statusText);return data;}
function excluded(){return [...document.querySelectorAll('.ablations input:checked')].map(x=>x.value);}
$('#promptForm').onsubmit=async e=>{e.preventDefault();if($('#send').disabled)return;let text=$('#prompt').value.trim()||(mode==='world'?'Run survival episode':'');if(!text)return;const activeMode=mode,activePlanner=$('#planner').value;message('user',text);$('#send').disabled=true;$('#activity').textContent=activeMode==='chat'?'Loading / thinking locally…':'Running '+activeMode+'…';try{let data=await request('/api/respond',{prompt:text,mode:activeMode,history,seed:Number($('#seed').value),disabled_carriers:excluded(),planner:activePlanner});message('assistant',data.text,data.backend);if(activeMode==='chat'){history.push({role:'user',content:text},{role:'assistant',content:data.text});history=history.slice(-6);}if(data.links)showArtifact(data);if(data.episode)showWorld(data.episode,data.backend);if(data.prediction?.routing)showRouting(data.prediction.routing);$('#activity').textContent='Completed locally';$('#prompt').value='';}catch(error){message('error',error.message);$('#activity').textContent='Request failed — see message';}finally{$('#send').disabled=false;refreshStatus();}};
$('#prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('#promptForm').requestSubmit();}});
function showRouting(weights){document.querySelectorAll('#routing>div').forEach((row,i)=>{const value=Math.max(0,Math.min(1,Number(weights[i])||0));row.querySelector('b').style.width=value*100+'%';row.querySelector('small').textContent=Math.round(value*100)+'%';});}
function stopWorld(){playing=false;if(animationHandle!==null){cancelAnimationFrame(animationHandle);animationHandle=null;}}
function showArtifact(data){stopWorld();$('#timeline').hidden=true;$('#visual').replaceChildren();let el;if(data.kind==='video'&&data.links.mp4){el=document.createElement('video');el.controls=true;el.loop=true;el.src=data.links.mp4;}else{el=document.createElement('img');el.src=data.links.gif||data.links.preview;el.alt='Generated '+data.scene.color+' '+data.scene.shape+' scene';}$('#visual').append(el);$('#artifactType').textContent=data.kind.toUpperCase();$('#artifactInfo').textContent=data.text+' '+(data.triangle_count?data.triangle_count+' triangles.':'');$('#downloads').replaceChildren();for(const [name,url]of Object.entries(data.links)){if(name==='preview'||name==='mtl')continue;let a=document.createElement('a');a.href=url;a.download='';a.textContent=name.toUpperCase()+' ↓';$('#downloads').append(a);}}
function drawIdle(){const c=$('#worldCanvas');if(!c)return;const ctx=c.getContext('2d');ctx.fillStyle='#0a1720';ctx.fillRect(0,0,c.width,c.height);ctx.strokeStyle='#23413f';for(let i=0;i<7;i++){ctx.beginPath();ctx.ellipse(300,175,55+i*21,25+i*11,-.24,0,Math.PI*2);ctx.stroke();}ctx.fillStyle='#8be4ca';ctx.beginPath();ctx.arc(356,150,5,0,Math.PI*2);ctx.fill();}
function showWorld(data,backend){stopWorld();episode=data;frame=0;lastFrame=0;playing=Array.isArray(data.trajectory)&&data.trajectory.length>0;$('#visual').replaceChildren();const c=document.createElement('canvas');c.id='worldCanvas';c.width=600;c.height=400;c.setAttribute('aria-label','Synthetic survival world');$('#visual').append(c);$('#timeline').hidden=!playing;$('#scrub').max=Math.max(0,(data.trajectory?.length||0)-1);$('#artifactType').textContent='TIDEPOOL';$('#artifactInfo').textContent=(data.survived?'Horizon reached alive':'Died: '+data.death_reason)+' · '+data.steps+' steps · seed '+data.seed+' · '+(backend||data.backend||controllerName(data.controller));$('#downloads').replaceChildren();if(playing)animateWorld();else drawIdle();}
function drawWorld(){if(!episode)return;const ctx=$('#worldCanvas').getContext('2d'),row=episode.trajectory[frame],state=row?.state||{},snap=episode.initial_snapshot;ctx.fillStyle='#091721';ctx.fillRect(0,0,600,400);const size=44,ox=38,oy=55;for(let y=0;y<6;y++)for(let x=0;x<6;x++){const patch=snap.patches[y*6+x],alpha=.14+.35*patch.food;ctx.fillStyle=`rgba(103,197,156,${alpha})`;ctx.fillRect(ox+x*size,oy+y*size,size-4,size-4);ctx.fillStyle='#5b9cae';ctx.fillRect(ox+x*size+4,oy+y*size+size-10,(size-12)*patch.water,3);}const x=state.x??state.position?.[0]??0,y=state.y??state.position?.[1]??0;ctx.fillStyle='#f2b68e';ctx.beginPath();ctx.arc(ox+x*size+20,oy+y*size+20,10,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#ffe0b9';ctx.lineWidth=2;ctx.stroke();ctx.font='12px Segoe UI';ctx.fillStyle='#a3b8c5';ctx.fillText('TIDEPOOL / SEEDED PATCHES',38,31);ctx.fillText('Initial resource map • agent position replays',38,345);const names=['Health','Energy','Hydration','Stamina'];names.forEach((name,i)=>{let value=row?.next_observation?.[i]??1;ctx.fillStyle='#b9cdd2';ctx.fillText(name,350,85+i*55);ctx.fillStyle='#233741';ctx.fillRect(350,96+i*55,200,6);ctx.fillStyle=value<.25?'#eea276':'#8be4ca';ctx.fillRect(350,96+i*55,200*Math.max(0,Math.min(1,value)),6);});ctx.fillStyle='#eea276';ctx.fillText((row?.action_name||'').toUpperCase()+' / STEP '+(frame+1),350,329);$('#scrub').value=frame;$('#tick').textContent=frame+1;}
let lastDecisionEvent=null;
function inspectDecision(event){
  if(event===lastDecisionEvent)return;
  lastDecisionEvent=event;
  const decision=event?.decision,candidates=decision?.candidates;
  $('#decisionPanel').hidden=!Array.isArray(candidates);
  if(!Array.isArray(candidates))return;
  const names=candidates.map(row=>row.action_name);
  const reason=decision.fallback_reason?decision.fallback_reason.replaceAll('_',' '):decision.override_accepted?'empirical advantage cleared the margin':'incumbent action retained';
  $('#decisionMeta').textContent='Step '+(frame+1)+' · executed '+names[decision.action]+' · policy '+names[decision.policy_action]+' · proposed '+names[decision.proposed_action??decision.action]+' · '+reason;
  $('#decisionCandidates').replaceChildren();
  for(const row of candidates){
    const tr=document.createElement('tr');
    if(row.action===decision.action)tr.className='atlas-row';
    const label=document.createElement('th');label.scope='row';label.textContent=row.action_name+(row.action===decision.action?' · executed':'');tr.append(label);
    const values=[...row.predicted_observation.slice(0,3).map(x=>(x*100).toFixed(1)+'%'),row.error_radius.toFixed(3),row.support_distance.toFixed(3),row.trusted?'Yes':'No'];
    for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}
    $('#decisionCandidates').append(tr);
  }
  $('#decisionSources').textContent=candidates.map(row=>row.action_name+':\n'+(row.source_ids.length?row.source_ids.join('\n'):'memory erased')).join('\n\n');
}
const drawWorldScene=drawWorld;
drawWorld=function(){drawWorldScene();inspectDecision(episode?.trajectory?.[frame]);};
let lastFrame=0;function animateWorld(time=0){animationHandle=null;if(!playing||!episode)return;if(time-lastFrame>70){frame=Math.min(frame+1,episode.trajectory.length-1);lastFrame=time;if(frame===episode.trajectory.length-1)playing=false;}drawWorld();if(playing)animationHandle=requestAnimationFrame(animateWorld);}
$('#scrub').oninput=()=>{stopWorld();frame=Number($('#scrub').value);drawWorld();};$('#replay').onclick=()=>{if(!episode?.trajectory?.length)return;stopWorld();frame=0;lastFrame=0;playing=true;animateWorld();};
$('#remember').onclick=async()=>{try{const r=await request('/api/remember',{text:$('#memoryText').value,carrier:$('#carrier').value});$('#memoryStatus').textContent=r.saved?'Fact saved locally.':'Not saved';$('#memoryText').value='';}catch(e){$('#memoryStatus').textContent=e.message;}};
$('#recall').onclick=async()=>{try{const r=await request('/api/respond',{prompt:$('#prompt').value||'memory',mode:'memory',disabled_carriers:excluded()});$('#memoryStatus').textContent=r.text;}catch(e){$('#memoryStatus').textContent=e.message;}};
function setAtlasStatus(atlas){
  atlasReady=atlas?.ready===true;
  $('#atlasState').textContent=atlasReady?'OPT-IN CANDIDATE':'ATLAS NOT READY';
  $('#atlasState').classList.toggle('available',atlasReady);
  $('#atlasNote').textContent=atlasReady?'A fitted Atlas candidate is available. The learned policy remains the default; availability does not establish better performance.':'The paired experiment needs a fitted Atlas candidate. '+(atlas?.error||'Prepare it with the local command below.');
  $('#atlasSetup').hidden=atlasReady;
  const option=$('#planner option[value="atlas"]');
  option.disabled=!atlasReady;
  option.textContent=atlasReady?'Counterfactual Atlas · experimental':'Counterfactual Atlas · unavailable';
  $('#runExperiment').disabled=experimentBusy||!atlasReady;
  if(!experimentBusy&&experimentOutcome==='idle')$('#experimentStatus').textContent=atlasReady?'Ready. Six controllers will run on the same local synthetic episodes.':'Fit the Atlas before running a paired comparison.';
}
const numeric=value=>typeof value==='number'&&Number.isFinite(value);
const decimal=(value,digits=2)=>numeric(value)?value.toFixed(digits):'—';
const percentage=value=>numeric(value)?(value*100).toFixed(1)+'%':'—';
const signed=value=>numeric(value)?(value>0?'+':'')+value.toFixed(3):'—';
function resultRow(values,arm){
  const row=document.createElement('tr');
  if(arm==='atlas')row.className='atlas-row';
  values.forEach((value,i)=>{const cell=document.createElement(i===0?'th':'td');if(i===0)cell.scope='row';cell.textContent=String(value);row.append(cell);});
  return row;
}
function receiptURL(value){
  if(typeof value!=='string'||!value)return null;
  try{const url=new URL(value,location.origin);return url.origin===location.origin&&['http:','https:'].includes(url.protocol)?url.href:null;}catch{return null;}
}
function showExperiment(data){
  currentExperiment=data;
  const summary=$('#experimentSummary'),paired=$('#experimentPaired');
  summary.replaceChildren();paired.replaceChildren();
  for(const [arm,row]of Object.entries(data.summary||{})){
    summary.append(resultRow([controllerName(arm),numeric(row.episodes)?row.episodes:'—',percentage(row.survival_rate),decimal(row.mean_steps,1),decimal(row.mean_reward,3),decimal(row.mean_latency_ms,2),percentage(row.fallback_rate)],arm));
  }
  for(const [arm,row]of Object.entries(data.paired||{})){
    if(!row||typeof row!=='object'||Array.isArray(row))continue;
    const interval=Array.isArray(row.ci95)&&row.ci95.length===2?'['+signed(row.ci95[0])+', '+signed(row.ci95[1])+']':'—';
    paired.append(resultRow([controllerName(arm),signed(row.mean_reward_delta),interval,[row.wins,row.ties,row.losses].map(x=>numeric(x)?x:'—').join(' / ')],arm));
  }
  $('#pairedHeading').textContent='Paired reward differences vs learned policy';
  $('#pairedCaption').textContent='Each controller minus the policy on the same seed; intervals are reported by the experiment.';
  $('#experimentIdentity').textContent='Experiment '+String(data.experiment_id||'receipt available');
  const url=receiptURL(data.artifact_url);
  $('#experimentReceipt').hidden=!url;
  if(url)$('#experimentReceipt').href=url;else $('#experimentReceipt').removeAttribute('href');
  $('#experimentReplay').hidden=!data.replay?.trajectory?.length;
  const limits=$('#experimentLimits');limits.replaceChildren();
  const items=Array.isArray(data.limits)?data.limits:data.limits&&typeof data.limits==='object'?Object.entries(data.limits).map(([key,value])=>key+': '+String(value)):data.limits?[String(data.limits)]:[];
  for(const value of items){const item=document.createElement('li');item.textContent=typeof value==='string'?value:JSON.stringify(value);limits.append(item);}
  $('#experimentResults').hidden=false;
}
$('#experimentForm').onsubmit=async event=>{
  event.preventDefault();if(experimentBusy||!atlasReady)return;
  const payload={seed:Number($('#experimentSeed').value),scarcity:Number($('#scarcity').value),episodes:Number($('#episodes').value),max_steps:Number($('#maxSteps').value)};
  experimentBusy=true;currentExperiment=null;experimentOutcome='running';$('#experimentResults').hidden=true;
  $('#experimentPanel').setAttribute('aria-busy','true');
  $('#experimentStatus').classList.remove('error');
  $('#experimentStatus').textContent='Running '+payload.episodes+' paired episodes per controller, up to '+payload.max_steps+' steps each. Local CPU work may take a while…';
  $('#runExperiment').textContent='Experiment running…';
  $('#experimentForm').querySelectorAll('input,button').forEach(control=>control.disabled=true);
  try{
    const data=await request('/api/experiment',payload);
    showExperiment(data);experimentOutcome='complete';
    $('#experimentStatus').textContent='Completed locally. Compare all controllers below; Atlas remains an opt-in candidate.';
    if(data.atlas)setAtlasStatus(data.atlas);
  }catch(error){experimentOutcome='failed';$('#experimentStatus').textContent='Experiment failed: '+error.message;$('#experimentStatus').classList.add('error');}
  finally{
    experimentBusy=false;$('#experimentPanel').setAttribute('aria-busy','false');
    $('#runExperiment').textContent='Run paired experiment ↗';
    $('#experimentForm').querySelectorAll('input,button').forEach(control=>control.disabled=false);
    $('#runExperiment').disabled=!atlasReady;
  }
};
$('#experimentReplay').onclick=()=>{if(!currentExperiment?.replay)return;showWorld(currentExperiment.replay);$('#visual').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'center'});};
async function refreshStatus(){try{const response=await fetch('/api/status');if(!response.ok)throw Error(response.statusText);const data=await response.json();$('#connection').textContent='● Connected · on this PC';$('#coreState').textContent=data.core_ready?'Checkpoint available':'Training / preparing';const r=data.reports||{};let n=r.core?.examples_seen??r.core?.training_receipt?.examples_seen??r.core_recent?.at(-1)?.examples_seen;$('#exposures').textContent=n==null?'—':Number(n).toLocaleString();const counts=r.language_data?.counts;$('#corpus').textContent=counts?Object.values(counts).reduce((a,b)=>a+b,0).toLocaleString()+' rows':'Preparing';$('#evidence').textContent=JSON.stringify(r,null,2);setAtlasStatus(data.atlas);}catch{$('#connection').textContent='Offline · start the local server';}}
drawIdle();refreshStatus();setInterval(refreshStatus,15000);
