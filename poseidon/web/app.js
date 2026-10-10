const $=s=>document.querySelector(s);
let mode='chat',history=[],episode=null,frame=0,playing=false,first=true;
let animationHandle=null,atlasReady=false,contrastReady=false,horizonReady=false,odysseyReady=false,helmReady=false,odysseusReady=false,mcoReady=false,auraReady=true,tesseraReady=true,mnemorphReady=true,chimeraReady=true,experimentBusy=false,requestBusy=false,serverBusy=false,restartRequired=false,statusInFlight=false,currentExperiment=null,experimentOutcome='idle',protocolChosen=false,activeJobId=null,jobPollHandle=null;
const candidateStatus={atlas:null,contrast:null,horizon:null,odyssey:null,helm:null,odysseus:null,aura:null,tessera:null,mnemorph:null,chimera:null};
const plannerNotes={policy:'Direct actions from the trained policy. No teacher fallback.',mpc:'Plans actions using the learned dynamics head. Predicted outcomes can be inaccurate.',hybrid:'Combines the learned policy with dynamics-based planning.',chimera:'Unified CHIMERA controller: Causeway quantum superposition (Schrödinger non-Hermitian gain & Born collapse), 256-d HRR holographic associative concept memory, 3D Diamond Lattice entorhinal grid cells, and MOLT ecdysis with NexusFlow flux.',atlas:'Experimental Atlas v2 with retrieved residuals and absolute error bounds. Availability is not evidence of superiority.',contrast:'Experimental Contrast Atlas with selected channel corrections and calibrated paired action margins. Inspect its decisions in the replay; calibration does not guarantee safety.',horizon:'Experimental Horizon Atlas v4 with multi-horizon return advantage, depletion-trap guards and held-out calibration margins.',odyssey:'Experimental Odyssey Atlas v5 with episodic topological cognitive mapping, replenishment decay tracking and goal-oriented waypoint navigation.',helm:'Experimental Helm Critic v0.6 with history-conditioned return critique, reservoir memory and calibrated advantage gating.',odysseus:'Experimental Odysseus Navigator with empirical macro-action transition matrices, Bayesian replenishment LCB estimation and stamina rest guards.',aura:'Experimental AURA biomimetic controller with Central Complex 16-wedge ring attractor compass, sparse Kenyon cell neuropil arbiter, and ratified Tessera macro-actions.',risk_aware:'Experimental uncertainty-aware ensemble controller.'};
const controllerNames={policy:'Learned policy',chimera:'CHIMERA Super-Controller',chimera_causeway:'CHIMERA · Causeway Superposition',chimera_hologram:'CHIMERA · Holographic Memory',chimera_lattice:'CHIMERA · Diamond Lattice',metamorph:'METAMORPH Controller',molt:'MOLT Developmental Ecdysis',nexusflow:'NexusFlow Causal Flux',chronos:'Chronos Cyclic Sync',mpc:'Dynamics MPC',neural_mpc:'Neural MPC',hybrid:'Policy + dynamics',atlas:'Counterfactual Atlas v2',atlas_v2:'Counterfactual Atlas v2',atlas_no_memory:'Atlas v2 without memory',contrast:'Contrast Atlas',contrast_absolute:'Contrast · absolute bounds',contrast_no_memory:'Contrast · no memory',contrast_unfiltered:'Contrast · unfiltered',horizon:'Horizon Atlas v4',horizon_no_memory:'Horizon · no memory',horizon_ungated:'Horizon · ungated',odyssey:'Odyssey Atlas v5',odyssey_no_memory:'Odyssey · no memory',odyssey_ungated:'Odyssey · ungated',helm:'Helm Critic v0.6',selected:'Helm · selected',observation:'Helm · observation',no_innovation:'Helm · no innovation',yoked:'Helm · yoked',ungated:'Helm · ungated',odysseus:'Odysseus Navigator',odysseus_calibrated:'Odysseus · calibrated',odysseus_ungated:'Odysseus · ungated',odysseus_no_memory:'Odysseus · no memory',aura:'AURA Biomimetic Central Complex',aura_no_ring:'AURA · no ring compass',aura_no_neuropil:'AURA · no neuropil arbiter',tessera:'Tessera Macro Commons',mnemorph:'Mnemorph Archaeological Memory',heuristic:'Heuristic control',random:'Random control',risk_aware:'Uncertainty ensemble'};
const controllerName=value=>controllerNames[value]||String(value||'Unspecified controller');
const notes={chat:'Local pretrained language model. Answers can be wrong; arithmetic is checked by a separate exact solver.',math:'Exact rational arithmetic and linear equations. Use explicit multiplication: 3*x + 7 = 22. This result comes from a tool, not unaided neural reasoning.',image:'Learned scene attributes → software renderer. Describe cube, sphere, pyramid or cylinder; color; one to three objects; and small, medium or large.',video:'A learned scene becomes a 2.4-second animation. Motions: still, orbit, bounce, spin. Geometric animation, not photorealistic video.',mesh:'Create basic OBJ + MTL and glTF assets from the same learned scene representation. Up to three geometric objects.',world:'The trained policy chooses six actions in a deterministic synthetic world. Reaches the 256-step horizon or dies. No teacher fallback.'};
const examples={chat:'Ask Poseidon something…',math:'(17 + 5) * 3, or 3*x + 7 = 22',image:'Create two small cyan spheres with orbit motion.',video:'Make three medium purple cubes with bounce motion.',mesh:'Build one large yellow pyramid with still motion.',world:'Run the learned agent in TidePool'};
notes.world='Choose a controller, resource scarcity and episode horizon for a deterministic synthetic world. Replay the actual decisions. Both Atlas candidates are opt-in experiments.';
$('#planner').onchange=()=>{$('#plannerNote').textContent=plannerNotes[$('#planner').value]||'Experimental survival controller.';};
function setMode(value){mode=value;document.querySelectorAll('[data-mode]').forEach(b=>{b.classList.toggle('selected',b.dataset.mode===mode);b.setAttribute('aria-selected',String(b.dataset.mode===mode));});$('#modeNote').textContent=notes[mode];$('#prompt').placeholder=examples[mode];if($('#plannerControls'))$('#plannerControls').hidden=(mode!=='world');}
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{if(b.dataset.target)setMode(b.dataset.target);$('#prompt').value=b.dataset.prompt;$('#prompt').focus();});
function message(role,text,backend){if(first){$('#messages').replaceChildren();first=false;}let card=document.createElement('div');card.className='message '+role;let label=document.createElement('div');label.className='message-label';label.textContent=role==='user'?'YOU':role==='error'?'REQUEST ERROR':'POSEIDON';let body=document.createElement('div');body.className='message-body';body.textContent=text;card.append(label,body);if(backend){let meta=document.createElement('div');meta.className='message-backend';meta.textContent=backend;card.append(meta);}$('#messages').append(card);$('#messages').scrollTop=$('#messages').scrollHeight;}
async function request(url,payload){const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await response.json();if(!response.ok)throw Error(data.error||response.statusText);return data;}
function excluded(){return [...document.querySelectorAll('.ablations input:checked')].map(x=>x.value);}
function worldSettings(){const invalid=[$('#worldScarcity'),$('#worldHorizon')].find(control=>!control.checkValidity());if(invalid){invalid.reportValidity();$('#activity').textContent='Check the Survival settings';return null;}return {scarcity:Number($('#worldScarcity').value),max_steps:Number($('#worldHorizon').value)};}
$('#promptForm').onsubmit=async e=>{e.preventDefault();if($('#send').disabled)return;let text=$('#prompt').value.trim()||(mode==='world'?'Run survival episode':'');if(!text)return;const activeMode=mode,activePlanner=$('#planner').value,settings=activeMode==='world'?worldSettings():{};if(settings===null)return;message('user',text);requestBusy=true;syncControls();$('#activity').textContent=activeMode==='chat'?'Loading / thinking locally…':'Running '+activeMode+'…';try{let data=await request('/api/respond',{prompt:text,mode:activeMode,history,seed:Number($('#seed').value),disabled_carriers:excluded(),planner:activePlanner,...settings});message('assistant',data.text,data.backend);if(activeMode==='chat'){history.push({role:'user',content:text},{role:'assistant',content:data.text});history=history.slice(-6);}if(data.links)showArtifact(data);if(data.episode)showWorld(data.episode,data.backend);if(data.prediction?.routing)showRouting(data.prediction.routing);$('#activity').textContent='Completed locally';$('#prompt').value='';}catch(error){message('error',error.message);$('#activity').textContent='Request failed — see message';}finally{requestBusy=false;serverBusy=false;syncControls();refreshStatus();}};
$('#prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('#promptForm').requestSubmit();}});
function showRouting(weights){document.querySelectorAll('#routing>div').forEach((row,i)=>{const value=Math.max(0,Math.min(1,Number(weights[i])||0));row.querySelector('b').style.width=value*100+'%';row.querySelector('small').textContent=Math.round(value*100)+'%';});}
function stopWorld(){playing=false;if(animationHandle!==null){cancelAnimationFrame(animationHandle);animationHandle=null;}}
function showArtifact(data){stopWorld();$('#timeline').hidden=true;$('#decisionPanel').hidden=true;$('#visual').replaceChildren();let el;if(data.kind==='video'&&data.links.mp4){el=document.createElement('video');el.controls=true;el.loop=true;el.src=data.links.mp4;}else{el=document.createElement('img');el.src=data.links.gif||data.links.preview;el.alt='Generated '+data.scene.color+' '+data.scene.shape+' scene';}$('#visual').append(el);$('#artifactType').textContent=data.kind.toUpperCase();$('#artifactInfo').textContent=data.text+' '+(data.triangle_count?data.triangle_count+' triangles.':'');$('#downloads').replaceChildren();for(const [name,url]of Object.entries(data.links)){if(name==='preview'||name==='mtl')continue;let a=document.createElement('a');a.href=url;a.download='';a.textContent=name.toUpperCase()+' ↓';$('#downloads').append(a);}}
function drawIdle(){const c=$('#worldCanvas');if(!c)return;const ctx=c.getContext('2d');ctx.fillStyle='#0a1720';ctx.fillRect(0,0,c.width,c.height);ctx.strokeStyle='#23413f';for(let i=0;i<7;i++){ctx.beginPath();ctx.ellipse(300,175,55+i*21,25+i*11,-.24,0,Math.PI*2);ctx.stroke();}ctx.fillStyle='#8be4ca';ctx.beginPath();ctx.arc(356,150,5,0,Math.PI*2);ctx.fill();}
function showWorld(data,backend){stopWorld();episode={...data,backend:backend||data.backend};frame=0;lastFrame=0;lastDecisionEvent=null;$('#decisionPanel').hidden=true;playing=Array.isArray(data.trajectory)&&data.trajectory.length>0;$('#visual').replaceChildren();const c=document.createElement('canvas');c.id='worldCanvas';c.width=600;c.height=400;c.setAttribute('aria-label','Synthetic survival world');$('#visual').append(c);$('#timeline').hidden=!playing;$('#scrub').max=Math.max(0,(data.trajectory?.length||0)-1);$('#artifactType').textContent='TIDEPOOL';$('#artifactInfo').textContent=(data.survived?'Horizon reached alive':'Died: '+data.death_reason)+' · '+data.steps+' steps · seed '+data.seed+' · '+(backend||data.backend||controllerName(data.controller));$('#downloads').replaceChildren();if(playing)animateWorld();else{drawIdle();inspectDecision(null);}}
function drawWorld(){if(!episode)return;const ctx=$('#worldCanvas').getContext('2d'),row=episode.trajectory[frame],state=row?.state||{},snap=episode.initial_snapshot;ctx.fillStyle='#091721';ctx.fillRect(0,0,600,400);const size=44,ox=38,oy=55;for(let y=0;y<6;y++)for(let x=0;x<6;x++){const patch=snap.patches[y*6+x],alpha=.14+.35*patch.food;ctx.fillStyle=`rgba(103,197,156,${alpha})`;ctx.fillRect(ox+x*size,oy+y*size,size-4,size-4);ctx.fillStyle='#5b9cae';ctx.fillRect(ox+x*size+4,oy+y*size+size-10,(size-12)*patch.water,3);}const x=state.x??state.position?.[0]??0,y=state.y??state.position?.[1]??0;ctx.fillStyle='#f2b68e';ctx.beginPath();ctx.arc(ox+x*size+20,oy+y*size+20,10,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#ffe0b9';ctx.lineWidth=2;ctx.stroke();ctx.font='12px Segoe UI';ctx.fillStyle='#a3b8c5';ctx.fillText('TIDEPOOL / SEEDED PATCHES',38,31);ctx.fillText('Initial resource map • agent position replays',38,345);const names=['Health','Energy','Hydration','Stamina'];names.forEach((name,i)=>{let value=row?.next_observation?.[i]??1;ctx.fillStyle='#b9cdd2';ctx.fillText(name,350,85+i*55);ctx.fillStyle='#233741';ctx.fillRect(350,96+i*55,200,6);ctx.fillStyle=value<.25?'#eea276':'#8be4ca';ctx.fillRect(350,96+i*55,200*Math.max(0,Math.min(1,value)),6);});ctx.fillStyle='#eea276';ctx.fillText((row?.action_name||'').toUpperCase()+' / STEP '+(frame+1),350,329);if(row?.decision?.causeway){ctx.fillStyle='#6cd4ff';ctx.fillText('CHIMERA: ψ-entropy '+decimal(row.decision.causeway.von_neumann_entropy,3)+' · Coh '+decimal(row.decision.lattice_coherence,2)+' · Holo '+decimal(row.decision.holographic_capacity_load*100,1)+'%',38,365);}else if(row?.decision?.cognitive_map){ctx.fillStyle='#8be4ca';ctx.fillText('COG MAP: '+row.decision.cognitive_map.discovered_patches+' patches · '+row.decision.cognitive_map.discovered_edges+' edges'+(row.decision.best_waypoint?' · WP dist '+row.decision.best_waypoint.distance:''),38,365);}if(row?.decision?.pretransit_rest){ctx.fillStyle='#eea276';ctx.fillText('PRE-TRANSIT REST GUARD ACTIVE',38,385);}$('#scrub').value=frame;$('#tick').textContent=frame+1;}
let lastDecisionEvent=null;
const featureNames=['Health','Energy','Hydration','Stamina','Exposure','Threat','Food','Water','Shelter quality','Weather severity','Temperature','Daylight','Terrain difficulty','Resource scent','Last action','Episode progress'];
const actionNames=['rest','forage','drink','shelter','explore','flee'];
function readable(value){return typeof value==='string'?value.replaceAll('_',' '):value==null?'—':JSON.stringify(value);}
function evidenceValue(label,value){const box=document.createElement('div'),name=document.createElement('span'),detail=document.createElement('strong');name.textContent=label;detail.textContent=value;box.append(name,detail);$('#decisionEvidence').append(box);}
function inspectDecision(event){
  if(event===lastDecisionEvent)return;
  lastDecisionEvent=event;
  const decision=event?.decision,candidates=decision?.candidates;
  $('#decisionPanel').hidden=!Array.isArray(candidates);
  if(!Array.isArray(candidates))return;
  const returns=decision.backend==='horizon-atlas-v4'||decision.backend==='odyssey-atlas-v5';
  $('#decisionHeading').textContent=returns?'Return advantages at the replay step':'Candidate futures at the replay step';
  const candidateTable=$('#decisionCandidates').closest('table');
  candidateTable.querySelector('caption').textContent=returns?'Estimated multi-step reward advantages and local action filters. Empirical radii describe calibration; heuristic overrides can bypass the margin gate.':'One-step predictions and paired action advantage over the policy action. Calibration is descriptive and does not guarantee safety.';
  const headers=returns?['Action','Return advantage','Local filter','Policy probability','Distance','Fitted support','Advantage radius','Return margin','Executed','Reason']:['Action','Health','Energy','Hydration','Vital error radius','Distance','Within budget','Point advantage','Advantage radius','Paired margin'];
  candidateTable.querySelectorAll('thead th').forEach((cell,i)=>cell.textContent=headers[i]);
  const nameFor=action=>candidates.find(row=>row.action===action)?.action_name||actionNames[action]||'—';
  const reason=decision.fallback_reason?decision.fallback_reason.replaceAll('_',' '):decision.override_accepted?'empirical advantage cleared the margin':'incumbent action retained';
  $('#decisionMeta').textContent=controllerName(episode?.controller||episode?.planner||episode?.backend)+' · step '+(frame+1)+' · executed '+nameFor(decision.action)+' · policy '+nameFor(decision.policy_action)+' · proposed '+nameFor(decision.proposed_action??decision.action)+' · '+reason;
  $('#decisionEvidence').replaceChildren();
  evidenceValue('Decision gate',readable(decision.gate||'absolute vital bounds'));
  evidenceValue('Override accepted',decision.override_accepted===true?'Yes':'No');
  if(numeric(decision.point_advantage))evidenceValue('Proposed action advantage',signed(decision.point_advantage));
  if(numeric(decision.advantage_error_radius))evidenceValue('Paired calibration radius',decimal(decision.advantage_error_radius,4));
  if(numeric(decision.empirical_advantage_margin))evidenceValue('Paired empirical margin',signed(decision.empirical_advantage_margin));
  if(returns&&numeric(decision.margin))evidenceValue('Return margin',signed(decision.margin));
  $('#decisionCandidates').replaceChildren();
  const incumbent=candidates.find(row=>row.action===decision.policy_action);
  for(const row of candidates){
    const tr=document.createElement('tr');
    if(row.action===decision.action)tr.className='atlas-row';
    const label=document.createElement('th');label.scope='row';label.textContent=row.action_name+(row.action===decision.action?' · executed':'');tr.append(label);
    const prediction=Array.isArray(row.predicted_observation)?row.predicted_observation:[];
    const pointAdvantage=numeric(row.point_advantage)?row.point_advantage:numeric(row.point_value)&&numeric(incumbent?.point_value)?row.point_value-incumbent.point_value:undefined;
    const proposed=row.action===(decision.proposed_action??decision.action);
    const radius=numeric(row.advantage_error_radius)?row.advantage_error_radius:proposed?decision.advantage_error_radius:undefined;
    const margin=numeric(row.empirical_advantage_margin)?row.empirical_advantage_margin:proposed?decision.empirical_advantage_margin:undefined;
    const values=returns?[signed(row.advantage),row.safe===true?'Pass':'Reject',percentage(row.policy_probability),decimal(decision.support_distance,3),decision.supported===true?'Yes':'No',decimal(decision.advantage_error_radius,4),proposed?signed(decision.margin):'—',row.action===decision.action?'Yes':'No',row.action===decision.action?readable(decision.fallback_reason):'—']:[...Array.from({length:3},(_,i)=>percentage(prediction[i])),decimal(row.error_radius,4),decimal(row.support_distance,3),row.trusted===true?'Yes':'No',signed(pointAdvantage),decimal(radius,4),signed(margin)];
    for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}
    $('#decisionCandidates').append(tr);
  }
  const executed=candidates.find(row=>row.action===decision.action);
  const weights=decision.feature_alphas||executed?.feature_alphas;
  $('#featureDetails').hidden=!Array.isArray(weights);
  $('#decisionFeatures').replaceChildren();
  if(Array.isArray(weights))weights.forEach((alpha,i)=>$('#decisionFeatures').append(resultRow([featureNames[i]||'Channel '+i,decimal(executed?.base_observation?.[i],4),decimal(executed?.predicted_observation?.[i],4),decimal(alpha,2)])));
  $('#decisionSources').textContent=candidates.map(row=>row.action_name+':\n'+(Array.isArray(row.source_ids)&&row.source_ids.length?row.source_ids.join('\n'):'no retrieved identities')).join('\n\n');
}
const drawWorldScene=drawWorld;
drawWorld=function(){drawWorldScene();inspectDecision(episode?.trajectory?.[frame]);};
let lastFrame=0;function animateWorld(time=0){animationHandle=null;if(!playing||!episode)return;if(time-lastFrame>70){frame=Math.min(frame+1,episode.trajectory.length-1);lastFrame=time;if(frame===episode.trajectory.length-1)playing=false;}drawWorld();if(playing)animationHandle=requestAnimationFrame(animateWorld);}
$('#scrub').oninput=()=>{stopWorld();frame=Number($('#scrub').value);drawWorld();};$('#replay').onclick=()=>{if(!episode?.trajectory?.length)return;stopWorld();frame=0;lastFrame=0;playing=true;animateWorld();};
$('#remember').onclick=async()=>{if($('#remember').disabled)return;requestBusy=true;syncControls();try{const r=await request('/api/remember',{text:$('#memoryText').value,carrier:$('#carrier').value});$('#memoryStatus').textContent=r.saved?'Fact saved locally.':'Not saved';$('#memoryText').value='';}catch(e){$('#memoryStatus').textContent=e.message;}finally{requestBusy=false;serverBusy=false;syncControls();refreshStatus();}};
$('#recall').onclick=async()=>{if($('#recall').disabled)return;requestBusy=true;syncControls();try{const r=await request('/api/respond',{prompt:$('#prompt').value||'memory',mode:'memory',disabled_carriers:excluded()});$('#memoryStatus').textContent=r.text;}catch(e){$('#memoryStatus').textContent=e.message;}finally{requestBusy=false;serverBusy=false;syncControls();refreshStatus();}};
function selectedProtocol(){return $('#experimentType').value||'hyperion';}
function protocolReady(){const p=selectedProtocol();return (p==='hyperion'||p==='chimera'||p==='metamorph'||p==='aura'||p==='tessera'||p==='mnemorph'||p==='mco')?true:p==='odysseus'?odysseusReady:p==='helm'?helmReady:p==='odyssey'?odysseyReady&&horizonReady&&contrastReady&&atlasReady:p==='horizon'?horizonReady&&contrastReady&&atlasReady:p==='contrast'?contrastReady&&atlasReady:atlasReady;}
let seedChosen=false;
$('#experimentSeed').addEventListener('input',()=>seedChosen=true);
$('#experimentType').onchange=()=>{protocolChosen=true;experimentOutcome='idle';updateProtocol();};
function syncControls(){
  const busy=experimentBusy||requestBusy||serverBusy;
  $('#send').disabled=busy;
  $('#remember').disabled=busy;
  $('#recall').disabled=busy;
  $('#planner').disabled=busy;
  $('#worldScarcity').disabled=busy;
  $('#worldHorizon').disabled=busy;
  $('#runExperiment').disabled=busy||restartRequired||!protocolReady();
  $('#experimentForm').querySelectorAll('input,select').forEach(control=>control.disabled=experimentBusy);
  $('#runExperiment').textContent=experimentBusy?'Experiment running…':restartRequired?'Server restart required':serverBusy?'Local server busy…':requestBusy?'Workspace request running…':'Run paired experiment ↗';
}
function updateProtocol(){
  const protocol=selectedProtocol();
  if(!seedChosen)$('#experimentSeed').value=protocol==='hyperion'?'303000001':protocol==='chimera'?'202000001':protocol==='metamorph'?'199000001':protocol==='aura'?'150000001':protocol==='tessera'?'160000001':protocol==='mnemorph'?'170000001':protocol==='odysseus'?'133000001':protocol==='mco'?'140000001':protocol==='helm'?'123000001':protocol==='odyssey'?'110000001':protocol==='horizon'?'109000001':protocol==='contrast'?'104000001':'93000001';
  $('#protocolNote').textContent=protocol==='hyperion'?'HYPERION frontier super-controller: Morpheus dream consolidation & hippocampal replay, Prometheus meta-plasticity & homeostatic allostasis, and NTAG intermittent RF energy harvesting on identical worlds.':protocol==='chimera'?'CHIMERA super-controller: Causeway quantum superposition (Schrödinger gain & Born collapse), 256-d HRR holographic associative memory, and 3D Diamond Lattice geodesic grid cells on identical worlds.':protocol==='metamorph'?'METAMORPH developmental life-stage ecdysis (MOLT), NexusFlow causal potential flux, and Chronos cyclic beaconing on identical worlds.':protocol==='aura'?'Biomimetic ring attractor heading tracking and sparse Kenyon cell neuropil arbitration on identical worlds.':protocol==='tessera'?'Independent-lineage verified macro-action commons and finite-domain semantic safety pre-checks on identical seeds.':protocol==='mnemorph'?'Mnemorph archaeological memory assays: hub vs periphery structural lesions, context shift, and associative regrowth.':protocol==='odysseus'?'Six paired arms evaluate Odysseus calibrated routing, ungated, no-memory, Odyssey, random against the frozen policy baseline.':protocol==='mco'?'Full causal factorial evaluation across 16 carrier combinations and 3 falsifiable negative controls on standard knowledge benchmarks.':protocol==='helm'?'Six paired arms compare Helm selected, observation-only, no-innovation, yoked and ungated critics against the frozen policy baseline.':protocol==='odyssey'?'Ten paired controllers compare Odyssey cognitive mapping against Horizon Atlas, Contrast Atlas, Atlas v2, Neural MPC and policy.':protocol==='horizon'?'Nine paired controllers separate multi-horizon return advantages, depletion-trap guards, memory removal and uncalibrated interventions on identical worlds.':protocol==='contrast'?'Nine paired controllers separate action advantage, absolute bounds, memory removal and unfiltered corrections. Atlas v2 is retained as a baseline.':'Six paired controllers compare the original Atlas v2 with memory removed, neural MPC, policy, heuristic and random controls.';
  $('#hyperionSetup').hidden=(protocol!=='hyperion');
  $('#chimeraSetup').hidden=(protocol!=='chimera');
  $('#metamorphSetup').hidden=(protocol!=='metamorph');
  $('#auraSetup').hidden=(protocol!=='aura');
  $('#tesseraSetup').hidden=(protocol!=='tessera');
  $('#mnemorphSetup').hidden=(protocol!=='mnemorph');
  $('#odysseusSetup').hidden=!(protocol==='odysseus'&&!odysseusReady);
  $('#mcoSetup').hidden=!(protocol==='mco'&&!mcoReady);
  $('#helmSetup').hidden=!(protocol==='helm'&&!helmReady);
  $('#odysseySetup').hidden=!(protocol==='odyssey'&&!odysseyReady);
  $('#horizonSetup').hidden=!(protocol==='horizon'&&!horizonReady);
  $('#contrastSetup').hidden=!(protocol==='contrast'&&!contrastReady);
  $('#atlasSetup').hidden=atlasReady;
  $('#jobPanel').hidden=false;
  if(!experimentBusy&&experimentOutcome==='idle')$('#experimentStatus').textContent=restartRequired?'Restart the local server to use the current source before recording an experiment.':serverBusy?'The server is handling another request. Latest availability comes from its cached status.':protocolReady()?'Ready. Compare controllers or benchmark factorial subsets on identical seeds.':'Prepare candidate dependencies before running this comparison.';
  syncControls();
}
function setCandidateStatus(kind,status){
  candidateStatus[kind]=status;
  const ready=status?.ready===true,label=kind==='hyperion'?'HYPERION Frontier Super-Controller':kind==='chimera'?'CHIMERA Super-Controller':kind==='aura'?'AURA Biomimetic Central Complex':kind==='tessera'?'Tessera Macro-Commons':kind==='mnemorph'?'Mnemorph Archaeological Memory':kind==='odysseus'?'Odysseus Navigator':kind==='helm'?'Helm Critic v0.6':kind==='odyssey'?'Odyssey Atlas v5':kind==='horizon'?'Horizon Atlas v4':kind==='contrast'?'Contrast Atlas':'Atlas v2';
  if(kind==='hyperion')hyperionReady=ready;else if(kind==='chimera')chimeraReady=ready;else if(kind==='aura')auraReady=ready;else if(kind==='tessera')tesseraReady=ready;else if(kind==='mnemorph')mnemorphReady=ready;else if(kind==='odysseus')odysseusReady=ready;else if(kind==='helm')helmReady=ready;else if(kind==='odyssey')odysseyReady=ready;else if(kind==='horizon')horizonReady=ready;else if(kind==='contrast')contrastReady=ready;else atlasReady=ready;
  if($('#'+kind+'State')){$('#'+kind+'State').textContent=ready?'OPT-IN CANDIDATE':'FIT NOT READY';$('#'+kind+'State').classList.toggle('available',ready);}
  const receipt=status?.fit_receipt,selection=receipt?.selection,calibration=receipt?.calibration;
  const sampleCount=receipt?.training_samples;
  let detail=ready?'A fitted candidate is available. ':label+' is unavailable. '+(status?.error||'Fit it with the local command below.')+' ';
  if(ready&&kind==='hyperion')detail='Morpheus dream consolidation (SWS & REM), Prometheus meta-plasticity, and NTAG intermittent RF energy harvesting. ';
  if(ready&&kind==='chimera')detail='Causeway quantum superposition (6-d state vector), 256-d HRR holographic associative concept memory & 3D diamond geodesic neuropil. ';
  if(ready&&kind==='aura')detail='16-wedge ring attractor compass (PVA coherence) & 128 Kenyon cell sparse neuropil arbiter (k=8 WTA). ';
  if(ready&&kind==='tessera')detail='Independent dual-lineage ratified macro-actions with finite-domain safety assertions. ';
  if(ready&&kind==='mnemorph')detail='Archaeological memory assays: token hub/periphery lesions and associative regrowth. ';
  if(ready&&kind==='odysseus'&&numeric(status?.fit_receipt?.error_radius))detail+='Error radius: '+decimal(status.fit_receipt.error_radius,3)+'. ';
  if(ready&&numeric(sampleCount))detail+=sampleCount.toLocaleString()+' fitted '+(kind==='horizon'||kind==='helm'?'anchor states':'branch transitions')+'. ';
  if(ready&&kind==='helm'){
    const family=selection?.family;
    if(family)detail+='Selected family: '+family+'. ';
    const radii=calibration?.selected?.radii;
    if(radii)detail+='Radii: H4='+decimal(radii['4'],3)+', H16='+decimal(radii['16'],3)+'. ';
  }
  if(ready&&kind==='contrast'){
    const alphas=selection?.alphas||selection?.feature_alphas;
    const corrected=Array.isArray(alphas)?alphas.filter(alpha=>numeric(alpha)&&alpha>0).length:null;
    if(corrected!==null)detail+=corrected+' of 16 channels admit retrieved corrections. ';
    if(numeric(selection?.episode_count))detail+='Selection: '+selection.episode_count+' episodes. ';
    const paired=calibration?.paired?.memory;
    if(numeric(paired?.episode_count))detail+='Paired calibration: '+paired.episode_count+' episodes. ';
  }
  if(ready&&kind==='horizon'){
    if(numeric(calibration?.advantage_error_radius))detail+='Advantage radius: '+decimal(calibration.advantage_error_radius,3)+'. ';
    if(numeric(calibration?.episodes))detail+='Calibration: '+calibration.episodes+' episodes. ';
  }
  detail+='The policy remains the default; availability does not establish better performance.';
  $('#'+kind+'Note').textContent=detail;
  const option=$('#planner option[value="'+kind+'"]');
  if(option){
    option.disabled=!ready;
    option.textContent=label+' · '+(ready?'experimental':'unavailable');
    if(!ready&&$('#planner').value===kind){$('#planner').value='policy';$('#planner').onchange();}
  }
}
const numeric=value=>typeof value==='number'&&Number.isFinite(value);
const decimal=(value,digits=2)=>numeric(value)?value.toFixed(digits):'—';
const percentage=value=>numeric(value)?(value*100).toFixed(1)+'%':'—';
const signed=value=>numeric(value)?(value>0?'+':'')+value.toFixed(3):'—';
function resultRow(values,arm){
  const row=document.createElement('tr');
  if(arm==='atlas'||arm==='contrast')row.className='atlas-row';
  values.forEach((value,i)=>{const cell=document.createElement(i===0?'th':'td');if(i===0)cell.scope='row';cell.textContent=String(value);row.append(cell);});
  return row;
}
function receiptURL(value){
  if(typeof value!=='string'||!value)return null;
  try{const url=new URL(value,location.origin);return url.origin===location.origin&&['http:','https:'].includes(url.protocol)?url.href:null;}catch{return null;}
}
function showPredictionComparison(comparison){
  const available=comparison&&typeof comparison==='object'&&!Array.isArray(comparison);
  $('#predictionPanel').hidden=!available;
  $('#predictionSummary').replaceChildren();$('#predictionChannels').replaceChildren();
  if(!available)return;
  const labels={base:'Base neural dynamics',unfiltered:'Unfiltered memory correction',memory:'Selected memory correction'};
  for(const key of ['base','unfiltered','memory']){
    const item=comparison[key];if(!item||typeof item!=='object')continue;
    $('#predictionSummary').append(resultRow([labels[key],numeric(item.anchors)?item.anchors:'—',numeric(item.branches)?item.branches:'—',decimal(item.mse,6)]));
  }
  featureNames.forEach((name,i)=>$('#predictionChannels').append(resultRow([name,...['base','unfiltered','memory'].map(key=>decimal(comparison[key]?.per_channel_mse?.[i],6))])));
}
function showRiskCoverage(risk){
  const available=risk&&typeof risk==='object'&&!Array.isArray(risk);
  $('#riskPanel').hidden=!available;
  $('#riskSummary').replaceChildren();
  if(!available)return;
  for(const [groupName,rows]of Object.entries(risk)){
    if(!Array.isArray(rows))continue;
    for(const row of rows){
      const label=groupName==='regular'?'Regular schedule':'Override anchors';
      const coverage=percentage(row.coverage);
      const adverse=numeric(row.adverse_rate)?row.adverse+' ('+percentage(row.adverse_rate)+')':'0';
      const gain=numeric(row.mean_actual_advantage)?signed(row.mean_actual_advantage):'—';
      const exposure=row.no_exposure?'No exposure':row.exposure+' / '+row.eligible;
      $('#riskSummary').append(resultRow([label+' (H'+row.horizon+')',signed(row.threshold),row.eligible,row.exposure,coverage,adverse,gain,exposure]));
    }
  }
}
function showExperiment(data){
  currentExperiment=data;
  const returns=data.schema==='poseidon-horizon-experiment-v1'||data.schema==='poseidon-odyssey-experiment-v1'||data.schema==='poseidon-helm-experiment-v1';
  const summary=$('#experimentSummary'),paired=$('#experimentPaired');
  summary.replaceChildren();paired.replaceChildren();
  for(const [arm,row]of Object.entries(data.summary||{})){
    if(!row||typeof row!=='object'||Array.isArray(row))continue;
    const audit=row.counterfactual_audit;
    const outcomeRows=Array.isArray(data.rows)?data.rows.filter(item=>item.arm===arm):[];
    const count=numeric(row.overrides)?row.overrides:outcomeRows.length&&outcomeRows.every(item=>numeric(item.overrides))?outcomeRows.reduce((sum,item)=>sum+item.overrides,0):undefined;
    const rate=numeric(row.override_rate)?row.override_rate:numeric(row.mean_steps)&&row.mean_steps>0&&numeric(row.episodes)?count/(row.mean_steps*row.episodes):undefined;
    const overrides=numeric(count)?count+' / '+percentage(rate):percentage(rate);
    const branchGain=numeric(audit?.mean_actual_advantage_at_overrides)?audit.mean_actual_advantage_at_overrides:count?outcomeRows.reduce((sum,item)=>sum+(item.counterfactual_audit?.mean_actual_advantage_at_overrides??0)*(item.overrides??0),0)/count:undefined;
    const latency=returns&&row.mean_steps>0?row.mean_decision_ms/row.mean_steps:row.mean_decision_ms??row.mean_latency_ms;
    summary.append(resultRow([controllerName(arm),numeric(row.episodes)?row.episodes:'—',percentage(row.survival_rate),decimal(row.mean_steps,1),decimal(row.mean_reward,3),decimal(latency,2),percentage(row.fallback_rate),overrides,signed(branchGain),decimal(row.mean_prediction_mse,5),decimal(row.mean_discovered_patches,2)],arm));
  }
  for(const [arm,row]of Object.entries(data.paired||{})){
    if(!row||typeof row!=='object'||Array.isArray(row))continue;
    const interval=Array.isArray(row.ci95)&&row.ci95.length===2?'['+signed(row.ci95[0])+', '+signed(row.ci95[1])+']':'—';
    const outcomes=Array.isArray(row.per_seed)?row.per_seed.map(item=>item.reward_delta??item.reward_diff).filter(numeric):[];
    const counts=[row.wins??outcomes.filter(value=>value>1e-10).length,row.ties??outcomes.filter(value=>Math.abs(value)<=1e-10).length,row.losses??outcomes.filter(value=>value< -1e-10).length];
    const name=arm.replace(/_vs_policy$/,'');
    paired.append(resultRow([controllerName(name),signed(row.mean_reward_delta??row.mean_reward_diff),interval,counts.map(x=>numeric(x)?x:'—').join(' / ')],name));
  }
  $('#pairedHeading').textContent='Paired reward differences vs learned policy';
  $('#pairedCaption').textContent='Each controller minus policy on the same seed. '+(returns?'This protocol provides no uncertainty interval.':'Intervals are reported by the experiment.');
  showPredictionComparison(data.prediction_comparison);
  showRiskCoverage(data.risk_coverage);
  const verification=data.verification;
  const replayedEpisodes=verification?.episodes_replayed??verification?.episodes;
  $('#experimentIdentity').textContent='Experiment '+String(data.experiment_id||'receipt available')+(verification?.verified===true?' · independently replay verified':'')+(numeric(replayedEpisodes)?' · '+replayedEpisodes+' episodes':'');
  const url=receiptURL(data.artifact_url);
  $('#experimentReceipt').hidden=!url;
  if(url)$('#experimentReceipt').href=url;else $('#experimentReceipt').removeAttribute('href');
  $('#experimentReplay').hidden=!data.replay?.trajectory?.length;
  const limits=$('#experimentLimits');limits.replaceChildren();
  const items=Array.isArray(data.limits)?data.limits:data.limits&&typeof data.limits==='object'?Object.entries(data.limits).map(([key,value])=>key+': '+String(value)):data.limits?[String(data.limits)]:[];
  for(const value of items){const item=document.createElement('li');item.textContent=typeof value==='string'?value:JSON.stringify(value);limits.append(item);}
  if(returns){const item=document.createElement('li');item.textContent='Branch utility gain covers one-step reserve utility. Decision return scores estimate multi-step reward; these measures have different units.';limits.append(item);}
  $('#experimentResults').hidden=false;
}
async function refreshJobs(){
  try{
    const res=await fetch('/api/jobs');if(!res.ok)return;
    const data=await res.json(),list=$('#jobHistory');list.replaceChildren();
    for(const job of (data.jobs||[])){
      const li=document.createElement('li');
      li.textContent='Job '+job.id+' · '+job.state+' ('+(job.created_at||'')+(job.finished_at?' → '+job.finished_at:'')+')';
      if(job.result){
        const btn=document.createElement('button');btn.type='button';btn.className='micro';btn.textContent='View result';
        btn.onclick=()=>showExperiment(job.result);li.append(' ',btn);
      }
      list.append(li);
    }
  }catch{}
}
$('#refreshJobs').onclick=refreshJobs;
$('#cancelJob').onclick=async()=>{
  if(!activeJobId)return;
  try{await fetch('/api/jobs/'+activeJobId+'/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});}catch{}
};
$('#experimentForm').onsubmit=async event=>{
  event.preventDefault();if(experimentBusy||requestBusy||serverBusy||restartRequired||!protocolReady())return;
  if(!$('#experimentForm').checkValidity()){$('#experimentForm').reportValidity();return;}
  const protocol=selectedProtocol();
  const payload={kind:protocol==='hyperion'?'hyperion-experiment':protocol==='aura'?'aura-experiment':protocol==='tessera'?'tessera-experiment':protocol==='mnemorph'?'mnemorph-experiment':protocol==='helm'?'helm-experiment':protocol,seed:Number($('#experimentSeed').value),scarcity:Number($('#scarcity').value),episodes:Number($('#episodes').value),max_steps:Number($('#maxSteps').value)};
  stopWorld();
  experimentBusy=true;currentExperiment=null;experimentOutcome='running';$('#experimentResults').hidden=true;
  $('#decisionPanel').hidden=true;
  $('#experimentPanel').setAttribute('aria-busy','true');
  $('#experimentStatus').classList.remove('error');
  $('#experimentStatus').textContent='Running '+(protocol==='hyperion'?'five':protocol==='chimera'?'four':protocol==='aura'?'five':protocol==='atlas'||protocol==='helm'||protocol==='odysseus'?'six':protocol==='contrast'||protocol==='horizon'?'nine':'ten')+' controllers on '+payload.episodes+' paired episodes each, up to '+payload.max_steps+' steps. Local CPU work may take a while…';
  syncControls();
  if(protocol==='hyperion'||protocol==='chimera'||protocol==='aura'||protocol==='tessera'||protocol==='mnemorph'||protocol==='helm'||protocol==='odysseus'||protocol==='mco'){
    try{
      const payload={kind:protocol==='hyperion'?'hyperion-experiment':protocol==='chimera'?'chimera-experiment':protocol==='aura'?'aura-experiment':protocol==='tessera'?'tessera-experiment':protocol==='mnemorph'?'mnemorph-experiment':protocol==='mco'?'mco-experiment':protocol==='odysseus'?'odysseus-experiment':'helm-experiment',seed:Number($('#experimentSeed').value),scarcity:Number($('#scarcity').value),episodes:Number($('#episodes').value),max_steps:Number($('#maxSteps').value)};
      const jobRes=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      const jobData=await jobRes.json();
      if(!jobRes.ok)throw Error(jobData.error||jobRes.statusText);
      activeJobId=jobData.id;
      $('#cancelJob').hidden=false;$('#jobIdentity').textContent='Job '+activeJobId+' (queued/running)';
      const poll=async()=>{
        try{
          const r=await fetch('/api/jobs/'+activeJobId);
          if(!r.ok)return;
          const j=await r.json();
          $('#jobIdentity').textContent='Job '+j.id+' · '+j.state;
          const comp=j.progress?.completed??0,tot=j.progress?.total??1;
          $('#jobProgress').value=tot>0?comp/tot:0;
          $('#jobStatus').textContent=j.progress?.phase||j.state;
          if(j.state==='completed'){
            clearInterval(jobPollHandle);activeJobId=null;$('#cancelJob').hidden=true;
            showExperiment(j.result);experimentOutcome='complete';
            $('#experimentStatus').textContent='Completed locally via durable job. Compare controllers or benchmark factorial subsets below.';
            experimentBusy=false;serverBusy=false;$('#experimentPanel').setAttribute('aria-busy','false');syncControls();refreshStatus();refreshJobs();
          }else if(j.state==='failed'||j.state==='cancelled'){
            clearInterval(jobPollHandle);activeJobId=null;$('#cancelJob').hidden=true;
            experimentOutcome='failed';$('#experimentStatus').textContent='Job '+j.state+': '+(j.error?.message||'No detail');$('#experimentStatus').classList.add('error');
            experimentBusy=false;serverBusy=false;$('#experimentPanel').setAttribute('aria-busy','false');syncControls();refreshStatus();refreshJobs();
          }
        }catch{}
      };
      jobPollHandle=setInterval(poll,1000);
      return;
    }catch(error){
      activeJobId=null;$('#cancelJob').hidden=true;
      experimentOutcome='failed';$('#experimentStatus').textContent='Experiment failed: '+error.message;$('#experimentStatus').classList.add('error');
      experimentBusy=false;serverBusy=false;$('#experimentPanel').setAttribute('aria-busy','false');syncControls();refreshStatus();
      return;
    }
  }
  try{
    const url=protocol==='hyperion'?'/api/hyperion-experiment':protocol==='chimera'?'/api/chimera-experiment':protocol==='metamorph'?'/api/metamorph-experiment':protocol==='aura'?'/api/aura-experiment':protocol==='tessera'?'/api/tessera-experiment':protocol==='mnemorph'?'/api/mnemorph-experiment':protocol==='odysseus'?'/api/odysseus-experiment':protocol==='mco'?'/api/mco-experiment':protocol==='odyssey'?'/api/odyssey-experiment':protocol==='horizon'?'/api/horizon-experiment':protocol==='contrast'?'/api/contrast-experiment':'/api/experiment';
    const data=await request(url,payload);
    showExperiment(data);experimentOutcome='complete';
    $('#experimentStatus').textContent='Completed locally. Compare controllers or benchmark factorial subsets below.';
    if(data.hyperion)setCandidateStatus('hyperion',data.hyperion);
    if(data.chimera)setCandidateStatus('chimera',data.chimera);
    if(data.metamorph)setCandidateStatus('metamorph',data.metamorph);
    if(data.aura)setCandidateStatus('aura',data.aura);
    if(data.tessera)setCandidateStatus('tessera',data.tessera);
    if(data.mnemorph)setCandidateStatus('mnemorph',data.mnemorph);
    if(data.atlas)setCandidateStatus('atlas',data.atlas);
    if(data.contrast)setCandidateStatus('contrast',data.contrast);
    if(data.horizon)setCandidateStatus('horizon',data.horizon);
    if(data.odyssey)setCandidateStatus('odyssey',data.odyssey);
    if(data.helm)setCandidateStatus('helm',data.helm);
    if(data.odysseus)setCandidateStatus('odysseus',data.odysseus);
  }catch(error){experimentOutcome='failed';$('#experimentStatus').textContent='Experiment failed: '+error.message;$('#experimentStatus').classList.add('error');}
  finally{
    experimentBusy=false;serverBusy=false;$('#experimentPanel').setAttribute('aria-busy','false');syncControls();refreshStatus();
  }
};
$('#experimentReplay').onclick=()=>{if(!currentExperiment?.replay)return;showWorld(currentExperiment.replay);$('#visual').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'center'});};
async function refreshStatus(){
  if(statusInFlight)return;
  statusInFlight=true;
  try{
    const response=await fetch('/api/status');if(!response.ok)throw Error(response.statusText);
    const data=await response.json();serverBusy=data.busy===true||data.atlas?.busy===true||data.contrast?.busy===true||data.horizon?.busy===true||data.odyssey?.busy===true||data.helm?.busy===true||data.odysseus?.busy===true;restartRequired=data.restart_required===true||data.source_current===false;
    const statusCached=data.status_cached===true||data.atlas?.status_cached===true||data.contrast?.status_cached===true||data.horizon?.status_cached===true||data.odyssey?.status_cached===true||data.helm?.status_cached===true||data.odysseus?.status_cached===true;
    $('#connection').textContent='● Connected · '+(restartRequired?'restart needed':serverBusy?'server busy':'on this PC')+(statusCached?' · cached status':'');
    $('#restartBanner').hidden=!restartRequired;
    const changed=Array.isArray(data.changed_source_files)?data.changed_source_files.filter(value=>typeof value==='string'):[];
    $('#restartNote').textContent='The loaded code differs from the files on disk. Restart the local server before recording an experiment.'+(changed.length?' Changed source: '+changed.join(', ')+'.':'');
    $('#coreState').textContent=data.core_ready?'Checkpoint available':'Training / preparing';
    const r=data.reports||{},n=r.core?.examples_seen??r.core?.training_receipt?.examples_seen??r.core_recent?.at(-1)?.examples_seen;
    $('#exposures').textContent=n==null?'—':Number(n).toLocaleString();
    const counts=r.language_data?.counts;
    $('#corpus').textContent=counts?Object.values(counts).reduce((a,b)=>a+b,0).toLocaleString()+' rows':'Preparing';
    $('#evidence').textContent=JSON.stringify(r,null,2);
    setCandidateStatus('atlas',data.atlas);setCandidateStatus('contrast',data.contrast);setCandidateStatus('horizon',data.horizon);setCandidateStatus('odyssey',data.odyssey);setCandidateStatus('helm',data.helm);setCandidateStatus('odysseus',data.odysseus);
    setCandidateStatus('aura',data.aura);setCandidateStatus('tessera',data.tessera);setCandidateStatus('mnemorph',data.mnemorph);setCandidateStatus('metamorph',data.metamorph);setCandidateStatus('chimera',data.chimera);setCandidateStatus('hyperion',data.hyperion);
    if(data.mco?.ready)mcoReady=true;
    if(!protocolChosen){$('#experimentType').value='hyperion';protocolChosen=true;}
    updateProtocol();
  }catch{$('#connection').textContent='Offline · start the local server';}
  finally{statusInFlight=false;syncControls();}
}
drawIdle();refreshStatus();refreshJobs();setInterval(refreshStatus,15000);
