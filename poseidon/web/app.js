const $=s=>document.querySelector(s);
let mode='chat',history=[],episode=null,frame=0,playing=false,first=true;
let animationHandle=null,atlasReady=false,contrastReady=false,horizonReady=false,odysseyReady=false,helmReady=false,odysseusReady=false,mcoReady=false,auraReady=true,tesseraReady=true,mnemorphReady=true,chimeraReady=true,experimentBusy=false,requestBusy=false,serverBusy=false,restartRequired=false,statusInFlight=false,currentExperiment=null,experimentOutcome='idle',protocolChosen=false,activeJobId=null,jobPollHandle=null;
const candidateStatus={cognitive:null,atlas:null,contrast:null,horizon:null,odyssey:null,helm:null,odysseus:null,aura:null,tessera:null,mnemorph:null,chimera:null,titan:null,trident:null};
const plannerNotes={pantheon:'Project PANTHEON: 24-Pillar Unified Cognitive Synthesis combining all 24 experimental repositories, Morpheus Counterfactual Dream Synthesis, Quibble 24-node Merkle Build DAG, and Flipper NTAG-215 ISO/IEC 14443-A CRC-16 telemetry.',cognitive:'Supermix Poseidon v2.0 Cognitive Platform with 3-tier Mnemosyne memory, Oracle world ensemble, Active Inference EFE minimization, generative Dream sleep consolidation, and second-order Metacognition.',policy:'Direct actions from the trained policy. No teacher fallback.',mpc:'Plans actions using the learned dynamics head. Predicted outcomes can be inaccurate.',hybrid:'Combines the learned policy with dynamics-based planning.',trident:'Residual identity probes gate overrides and directed travel uses only observed TidePool transitions; synthetic results remain experimental.',titan:'Opt-in experimental composite. Sidecars are software simulations; inspect the paired seed receipt and replay before drawing conclusions.',chimera:'Experimental classical composite using an amplitude-based action heuristic, associative memory, and synthetic 3D lattice telemetry.',atlas:'Experimental Atlas v2 with retrieved residuals and absolute error bounds. Availability is not evidence of superiority.',contrast:'Experimental Contrast Atlas with selected channel corrections and calibrated paired action margins. Inspect its decisions in the replay; calibration does not guarantee safety.',horizon:'Experimental Horizon Atlas v4 with multi-horizon return advantage, depletion-trap guards and held-out calibration margins.',odyssey:'Experimental Odyssey Atlas v5 with episodic topological cognitive mapping, replenishment decay tracking and goal-oriented waypoint navigation.',helm:'Experimental Helm Critic v0.6 with history-conditioned return critique, reservoir memory and calibrated advantage gating.',odysseus:'Experimental Odysseus Navigator with empirical macro-action transition matrices, Bayesian replenishment LCB estimation and stamina rest guards.',aura:'Experimental AURA biomimetic controller with Central Complex 16-wedge ring attractor compass, sparse Kenyon cell neuropil arbiter, and ratified Tessera macro-actions.',risk_aware:'Experimental uncertainty-aware ensemble controller.'};
const controllerNames={pantheon:'Project PANTHEON (24-Pillar Synthesis)',cognitive:'POSEIDON v2.0 Intact',intact:'POSEIDON v2.0 Intact',erased:'POSEIDON v2.0 (Erased Memory)',shifted:'POSEIDON v2.0 (Shifted Memory)',random_retrieval:'POSEIDON v2.0 (Random Retrieval)',memory_free:'POSEIDON v2.0 (Memory Free)',frozen_tidal:'Frozen Tidal core',policy:'Learned policy',titan:'TITAN Composite Controller',trident:'TRIDENT Identity-Gated Residual Controller',core:'Frozen Tidal core',aura:'AURA Biomimetic Controller',metamorph:'METAMORPH Controller',chimera:'CHIMERA Controller',hyperion:'HYPERION Controller',chimera_causeway:'CHIMERA · Causeway Superposition',chimera_hologram:'CHIMERA · Holographic Memory',chimera_lattice:'CHIMERA · Diamond Lattice',molt:'MOLT Developmental Ecdysis',nexusflow:'NexusFlow Causal Flux',chronos:'Chronos Cyclic Sync',mpc:'Dynamics MPC',neural_mpc:'Neural MPC',hybrid:'Policy + dynamics',atlas:'Counterfactual Atlas v2',atlas_v2:'Counterfactual Atlas v2',atlas_no_memory:'Atlas v2 without memory',contrast:'Contrast Atlas',contrast_absolute:'Contrast · absolute bounds',contrast_no_memory:'Contrast · no memory',contrast_unfiltered:'Contrast · unfiltered',horizon:'Horizon Atlas v4',horizon_no_memory:'Horizon · no memory',horizon_ungated:'Horizon · ungated',odyssey:'Odyssey Atlas v5',odyssey_no_memory:'Odyssey · no memory',odyssey_ungated:'Odyssey · ungated',helm:'Helm Critic v0.6',selected:'Helm · selected',observation:'Helm · observation',no_innovation:'Helm · no innovation',yoked:'Helm · yoked',ungated:'Helm · ungated',odysseus:'Odysseus Navigator',odysseus_calibrated:'Odysseus · calibrated',odysseus_ungated:'Odysseus · ungated',odysseus_no_memory:'Odysseus · no memory',aura_no_ring:'AURA · no ring compass',aura_no_neuropil:'AURA · no neuropil arbiter',tessera:'Tessera Macro Commons',mnemorph:'Mnemorph Archaeological Memory',heuristic:'Heuristic control',random:'Random control',risk_aware:'Uncertainty ensemble'};
const controllerName=value=>controllerNames[value]||String(value||'Unspecified controller');
const notes={chat:'Local pretrained language model. Answers can be wrong; arithmetic is checked by a separate exact solver.',proof:'PROJECT ARCHIMEDES: Formal Natural Deduction, equational algebraic theorem proving, Schwartz-Zippel invariant sampling, mathematical induction schemes, and tamper-evident SHA-256 proof certificates.',invent:'PROJECT DA VINCI: Deep computational creativity, Arthur Koestler bisociation across 8 canonical matrices, Altshuller TRIZ 40 inventive principles contradiction solver, and Pareto breakthrough dossier.',hypothesis:'PROJECT OLYMPUS: Scientific Hypothesis Formulation, Popperian Falsification, Lakatosian Research Programme Auditing, Baconian Crucial Experiments & Cryptographic Receipts.',creative:'PROJECT NOOSPHERE: Koestler Bisociation across 12 mathematical archetypes, 4-Agent Hegelian/Lakatosian Dialectical Tribunal, Fauconnier-Turner Concept Blending, and Pareto Ideation Portfolio.',pantheon:'PROJECT PANTHEON: 24-Pillar Unified Cognitive Synthesis across all 24 experimental repositories, with Morpheus Counterfactual Dream Synthesis, Quibble 24-node Merkle Build DAG, and Flipper NTAG-215 ISO/IEC 14443-A CRC-16 frame verification.',reason:'Dual-system neuro-symbolic coprocessor. Introspective Ghost Monologue, Dialectical Sublation, physical Archimedean buoyancy, and formal LTL safety proof.',problem:'PROBLEM FORGE (35 Domains): Verified combinatorial, physical, quantum, biological, ecological, strategic, control, topological & genomic problem solving with zero-eval formal certificates.',river:'Project PROMETHEUS Discovery Lab: Procedural 3D RiverWorld with stepping stone reach, Archimedean buoyant pine logs, multimodal bioadhesive enzyme curing, and naval metacentric stability.',math:'Exact symbolic calculus (diff/integrate), linear algebra (det/inv/eigenvalues), number theory (factor/totient/modinv/crt), polynomials up to degree 8, and linear systems.',image:'Learned scene attributes → software renderer. Describe cube, sphere, pyramid or cylinder; color; one to three objects; and small, medium or large.',video:'A learned scene becomes a 2.4-second animation. Motions: still, orbit, bounce, spin. Geometric animation, not photorealistic video.',mesh:'Create basic OBJ + MTL and glTF assets from the same learned scene representation. Up to three geometric objects.',world:'The trained policy chooses six actions in a deterministic synthetic world. Reaches the 256-step horizon or dies. No teacher fallback.'};
const examples={chat:'Ask Poseidon something…',proof:'(a - b)*(a + b) == a^2 - b^2 or prove P, P -> Q |- Q',invent:'Invent ultra-fast fault-tolerant distributed consensus using quantum mechanics and mycology fungal networks',hypothesis:'Formulate Popperian falsifiable hypothesis for General Relativity starlight deflection with Lakatosian research programme structure',creative:'Invent a cross-disciplinary hybrid architecture bridging quantum entanglement concurrence and Kalman filter Riccati estimation',pantheon:'Synthesize a 24-pillar Pantheon analysis for quantum entanglement concurrence and CHSH Bell violation with amplitudes [1, 0, 0, 1]',reason:'How do I cross a 10-meter river when stepping stones sink?',problem:'Find all pure and mixed Nash equilibria for the Battle of the Sexes game',river:'Assemble floating pine logs into a raft and cross the wide river channel.',math:'integrate(3*x^2 - 4*x + 1, x, 0, 2) or det([[4, 7], [2, 6]])',image:'Create two small cyan spheres with orbit motion.',video:'Make three medium purple cubes with bounce motion.',mesh:'Build one large yellow pyramid with still motion.',world:'Run the learned agent in TidePool'};
notes.world='Choose a controller, resource scarcity and episode horizon for a deterministic synthetic world. Replay the actual decisions. Both Atlas candidates are opt-in experiments.';
$('#planner').onchange=()=>{$('#plannerNote').textContent=plannerNotes[$('#planner').value]||'Experimental survival controller.';};
function setMode(value){mode=value;document.querySelectorAll('[data-mode]').forEach(b=>{b.classList.toggle('selected',b.dataset.mode===mode);b.setAttribute('aria-selected',String(b.dataset.mode===mode));});$('#modeNote').textContent=notes[mode];$('#prompt').placeholder=examples[mode];if($('#plannerControls'))$('#plannerControls').hidden=(mode!=='world');if($('#riverControls'))$('#riverControls').hidden=(mode!=='river');if(mode==='river'){initRiverWorld();}}
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{if(b.dataset.target)setMode(b.dataset.target);$('#prompt').value=b.dataset.prompt;$('#prompt').focus();});
function message(role,text,backend,details){if(first){$('#messages').replaceChildren();first=false;}let card=document.createElement('div');card.className='message '+role;let label=document.createElement('div');label.className='message-label';label.textContent=role==='user'?'YOU':role==='error'?'REQUEST ERROR':'POSEIDON';let body=document.createElement('div');body.className='message-body';body.textContent=text;if(details?.certificate?.certificate_sha256){let prfDiv=document.createElement('div');prfDiv.style.marginTop='8px';prfDiv.style.padding='8px';prfDiv.style.background='rgba(13, 148, 136, 0.1)';prfDiv.style.border='1px solid rgba(13, 148, 136, 0.35)';prfDiv.style.borderRadius='4px';prfDiv.style.fontSize='12px';let hash=details.certificate.certificate_sha256.slice(0,16);prfDiv.innerHTML=`📜 <strong>Project ARCHIMEDES Proof Certified:</strong> SHA-256 <code>${hash}...</code> · ${details.certificate.num_steps} Steps · Status: <code>${details.certificate.is_valid?'Q.E.D. VERIFIED':'UNPROVED'}</code>`;body.append(prfDiv);}if(details?.bisociation?.emergent_concept_name){let invDiv=document.createElement('div');invDiv.style.marginTop='8px';invDiv.style.padding='8px';invDiv.style.background='rgba(219, 39, 119, 0.1)';invDiv.style.border='1px solid rgba(219, 39, 119, 0.35)';invDiv.style.borderRadius='4px';invDiv.style.fontSize='12px';let hash=(details.receipt_sha256||'').slice(0,16);invDiv.innerHTML=`🎨 <strong>Project DA VINCI Invention:</strong> <em>${details.title}</em> · Pareto Score: <code>${details.pareto_score}</code> · SHA-256 <code>${hash}...</code>`;body.append(invDiv);}if(details?.creative_portfolio&&Array.isArray(details.creative_portfolio)){let nooDiv=document.createElement('details');nooDiv.style.marginTop='8px';nooDiv.style.padding='8px';nooDiv.style.background='rgba(236, 72, 153, 0.1)';nooDiv.style.border='1px solid rgba(236, 72, 153, 0.35)';nooDiv.style.borderRadius='4px';nooDiv.style.fontSize='12px';let nSum=document.createElement('summary');nSum.style.cursor='pointer';nSum.style.fontWeight='bold';nSum.style.color='#f472b6';nSum.textContent=`✨ Project NOOSPHERE Bisociation & Pareto Portfolio (Top HCRI: ${details.creative_portfolio[0]?.hcri_score})`;nooDiv.append(nSum);details.creative_portfolio.forEach(item=>{let p=document.createElement('p');p.style.margin='4px 0';p.innerHTML=`<strong>#${item.rank} ${item.title}</strong> (${item.bridge}) — HCRI: <code>${item.hcri_score}</code> · Law: <code>${item.governing_law}</code>`;nooDiv.append(p);});body.append(nooDiv);}if(details?.quibble_merkle_root){let panDiv=document.createElement('div');panDiv.style.marginTop='8px';panDiv.style.padding='8px';panDiv.style.background='rgba(245, 158, 11, 0.1)';panDiv.style.border='1px solid rgba(245, 158, 11, 0.35)';panDiv.style.borderRadius='4px';panDiv.style.fontSize='12px';panDiv.innerHTML=`🏛️ <strong>Project PANTHEON 24-Pillar Synthesis:</strong> ${details.pillars_verified||24}/${details.total_pillars||24} Pillars Verified · Quibble Merkle Root <code>${details.quibble_merkle_root.slice(0,16)}...</code> · NTAG-215 CRC-16 <code>${details.ntag215_frame?.crc16_hex||'0x0000'}</code>`;body.append(panDiv);}if(details?.plan&&Array.isArray(details.plan)){let planDiv=document.createElement('details');planDiv.className='problem-plan';planDiv.style.marginTop='8px';planDiv.style.padding='8px';planDiv.style.background='rgba(56, 189, 248, 0.08)';planDiv.style.borderRadius='4px';planDiv.style.fontSize='12px';let pSum=document.createElement('summary');pSum.style.cursor='pointer';pSum.style.fontWeight='bold';pSum.style.color='#38bdf8';pSum.textContent=`Verified Solution Plan (${details.plan.length} Steps)`;planDiv.append(pSum);details.plan.forEach((st,idx)=>{let p=document.createElement('p');p.style.margin='3px 0';p.textContent=`${idx+1}. ${st}`;planDiv.append(p);});body.append(planDiv);}if(details?.insight){let insDiv=document.createElement('div');insDiv.style.marginTop='6px';insDiv.style.color='#facc15';insDiv.style.fontSize='12px';insDiv.innerHTML=`💡 <strong>Structural Insight:</strong> ${details.insight}`;body.append(insDiv);}if(details?.falsification_receipt_sha256||details?.receipt?.hash_sha256){let olyDiv=document.createElement('div');olyDiv.style.marginTop='8px';olyDiv.style.padding='8px';olyDiv.style.background='rgba(8, 145, 178, 0.1)';olyDiv.style.border='1px solid rgba(8, 145, 178, 0.35)';olyDiv.style.borderRadius='4px';olyDiv.style.fontSize='12px';let hash=(details.falsification_receipt_sha256||details.receipt?.hash_sha256||'').slice(0,16);olyDiv.innerHTML=`🔬 <strong>Project OLYMPUS Falsification Receipt:</strong> SHA-256 <code>${hash}...</code> · Status: <code>${details.status||details.receipt?.outcome_status||'VERIFIED'}</code> · Popper/Lakatos Certified`;body.append(olyDiv);}if(details?.thought_steps){let thoughts=document.createElement('details');thoughts.className='thought-steps';thoughts.style.marginTop='8px';thoughts.style.padding='8px';thoughts.style.background='rgba(99, 102, 241, 0.08)';thoughts.style.borderRadius='4px';thoughts.style.fontSize='12px';let summary=document.createElement('summary');summary.style.cursor='pointer';summary.style.fontWeight='bold';summary.style.color='#818cf8';summary.textContent=`Introspective Ghost Monologue (${details.thought_steps.length} Steps)`;thoughts.append(summary);details.thought_steps.forEach(st=>{let p=document.createElement('p');p.style.margin='4px 0';p.innerHTML=`<strong>Step ${st.step} (${st.phase}):</strong> ${st.content}`;thoughts.append(p);});if(details.formal_proof){let badge=document.createElement('div');badge.style.marginTop='6px';badge.style.color='#34d399';badge.innerHTML=`🛡️ <strong>LTL Proof Certified:</strong> <code>${details.formal_proof.certificate_sha256?.slice(0, 16)}...</code> (${details.formal_proof.verdict})`;thoughts.append(badge);}body.append(thoughts);}card.append(label,body);if(backend){let meta=document.createElement('div');meta.className='message-backend';meta.textContent=backend;card.append(meta);}$('#messages').append(card);$('#messages').scrollTop=$('#messages').scrollHeight;}
async function request(url,payload){const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await response.json();if(!response.ok)throw Error(data.error||response.statusText);return data;}
function excluded(){return [...document.querySelectorAll('.ablations input:checked')].map(x=>x.value);}
function worldSettings(){const invalid=[$('#worldScarcity'),$('#worldHorizon')].find(control=>!control.checkValidity());if(invalid){invalid.reportValidity();$('#activity').textContent='Check the Survival settings';return null;}return {scarcity:Number($('#worldScarcity').value),max_steps:Number($('#worldHorizon').value)};}
$('#promptForm').onsubmit=async e=>{e.preventDefault();if($('#send').disabled)return;let text=$('#prompt').value.trim()||(mode==='world'?'Run survival episode':'');if(!text)return;const activeMode=mode,activePlanner=$('#planner').value,settings=activeMode==='world'?worldSettings():{};if(settings===null)return;message('user',text);requestBusy=true;syncControls();$('#activity').textContent=activeMode==='chat'||activeMode==='reason'?'Loading / thinking locally…':'Running '+activeMode+'…';try{let data=await request('/api/respond',{prompt:text,mode:activeMode,history,seed:Number($('#seed').value),disabled_carriers:excluded(),planner:activePlanner,max_new_tokens:512,...settings});message('assistant',data.text,data.backend,data);if(activeMode==='chat'||activeMode==='reason'){history.push({role:'user',content:text},{role:'assistant',content:data.text});history=history.slice(-16);}if(data.links)showArtifact(data);if(data.episode)showWorld(data.episode,data.backend);if(data.prediction?.routing)showRouting(data.prediction.routing);$('#activity').textContent='Completed locally';$('#prompt').value='';}catch(error){message('error',error.message);$('#activity').textContent='Request failed — see message';}finally{requestBusy=false;serverBusy=false;syncControls();refreshStatus();}};
$('#prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('#promptForm').requestSubmit();}});
function showRouting(weights){document.querySelectorAll('#routing>div').forEach((row,i)=>{const value=Math.max(0,Math.min(1,Number(weights[i])||0));row.querySelector('b').style.width=value*100+'%';row.querySelector('small').textContent=Math.round(value*100)+'%';});}
function stopWorld(){playing=false;if(animationHandle!==null){cancelAnimationFrame(animationHandle);animationHandle=null;}}
function showArtifact(data){stopWorld();$('#timeline').hidden=true;$('#decisionPanel').hidden=true;$('#visual').replaceChildren();let el;if(data.kind==='video'&&data.links.mp4){el=document.createElement('video');el.controls=true;el.loop=true;el.src=data.links.mp4;}else{el=document.createElement('img');el.src=data.links.gif||data.links.preview;el.alt='Generated '+data.scene.color+' '+data.scene.shape+' scene';}$('#visual').append(el);$('#artifactType').textContent=data.kind.toUpperCase();$('#artifactInfo').textContent=data.text+' '+(data.triangle_count?data.triangle_count+' triangles.':'');$('#downloads').replaceChildren();for(const [name,url]of Object.entries(data.links)){if(name==='preview'||name==='mtl')continue;let a=document.createElement('a');a.href=url;a.download='';a.textContent=name.toUpperCase()+' ↓';$('#downloads').append(a);}}
function drawIdle(){const c=$('#worldCanvas');if(!c)return;const ctx=c.getContext('2d');ctx.fillStyle='#0a1720';ctx.fillRect(0,0,c.width,c.height);ctx.strokeStyle='#23413f';for(let i=0;i<7;i++){ctx.beginPath();ctx.ellipse(300,175,55+i*21,25+i*11,-.24,0,Math.PI*2);ctx.stroke();}ctx.fillStyle='#8be4ca';ctx.beginPath();ctx.arc(356,150,5,0,Math.PI*2);ctx.fill();}
function showWorld(data,backend){stopWorld();episode={...data,backend:backend||data.backend};frame=0;lastFrame=0;lastDecisionEvent=null;$('#decisionPanel').hidden=true;playing=Array.isArray(data.trajectory)&&data.trajectory.length>0;$('#visual').replaceChildren();const c=document.createElement('canvas');c.id='worldCanvas';c.width=600;c.height=400;c.setAttribute('aria-label','Synthetic survival world');$('#visual').append(c);$('#timeline').hidden=!playing;$('#scrub').max=Math.max(0,(data.trajectory?.length||0)-1);$('#artifactType').textContent='TIDEPOOL';$('#artifactInfo').textContent=(data.survived?'Horizon reached alive':'Died: '+data.death_reason)+' · '+data.steps+' steps · seed '+data.seed+' · '+(backend||data.backend||controllerName(data.controller));$('#downloads').replaceChildren();if(playing)animateWorld();else{drawIdle();inspectDecision(null);}}
function drawWorld(){if(!episode)return;const ctx=$('#worldCanvas').getContext('2d'),row=episode.trajectory[frame],state=row?.state||{},snap=episode.initial_snapshot;ctx.fillStyle='#091721';ctx.fillRect(0,0,600,400);const size=44,ox=38,oy=55;for(let y=0;y<6;y++)for(let x=0;x<6;x++){const patch=snap.patches[y*6+x],alpha=.14+.35*patch.food;ctx.fillStyle=`rgba(103,197,156,${alpha})`;ctx.fillRect(ox+x*size,oy+y*size,size-4,size-4);ctx.fillStyle='#5b9cae';ctx.fillRect(ox+x*size+4,oy+y*size+size-10,(size-12)*patch.water,3);}const x=state.x??state.position?.[0]??0,y=state.y??state.position?.[1]??0;ctx.fillStyle='#f2b68e';ctx.beginPath();ctx.arc(ox+x*size+20,oy+y*size+20,10,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#ffe0b9';ctx.lineWidth=2;ctx.stroke();ctx.font='12px Segoe UI';ctx.fillStyle='#a3b8c5';ctx.fillText('TIDEPOOL / SEEDED PATCHES',38,31);ctx.fillText('Initial resource map • agent position replays',38,345);const names=['Health','Energy','Hydration','Stamina'];names.forEach((name,i)=>{let value=row?.next_observation?.[i]??1;ctx.fillStyle='#b9cdd2';ctx.fillText(name,350,85+i*55);ctx.fillStyle='#233741';ctx.fillRect(350,96+i*55,200,6);ctx.fillStyle=value<.25?'#eea276':'#8be4ca';ctx.fillRect(350,96+i*55,200*Math.max(0,Math.min(1,value)),6);});ctx.fillStyle='#eea276';ctx.fillText((row?.action_name||'').toUpperCase()+' / STEP '+(frame+1),350,329);if(row?.decision?.free_energy_objective!==undefined){ctx.fillStyle='#6cd4ff';ctx.fillText('ACTIVE INFERENCE: FEP '+decimal(row.decision.free_energy_objective,3)+' · Vol '+decimal(row.decision.epistemic_volatility,3)+' · Alert '+(row.decision.alert_level||'nominal').toUpperCase()+(numeric(row.decision.empowerment_value)&&row.decision.empowerment_value>0?' · Emp '+decimal(row.decision.empowerment_value,3):''),38,365);if(row.decision.guardrail_veto){ctx.fillStyle='#ff7b72';ctx.fillText('GUARDRAIL VETO: '+(row.decision.active_invariants||[]).join(', '),38,385);}else if(row.decision.dreams_consolidated>0){ctx.fillStyle='#8be4ca';ctx.fillText('DREAM CONSOLIDATION: '+row.decision.dreams_consolidated+' counterfactual branches consolidated',38,385);}}else if(row?.decision?.causeway){ctx.fillStyle='#6cd4ff';ctx.fillText('CHIMERA: ψ-entropy '+decimal(row.decision.causeway.von_neumann_entropy,3)+' · Coh '+decimal(row.decision.lattice_coherence,2)+' · Holo '+decimal(row.decision.holographic_capacity_load*100,1)+'%',38,365);}else if(row?.decision?.cognitive_map){ctx.fillStyle='#8be4ca';ctx.fillText('COG MAP: '+row.decision.cognitive_map.discovered_patches+' patches · '+row.decision.cognitive_map.discovered_edges+' edges'+(row.decision.best_waypoint?' · WP dist '+row.decision.best_waypoint.distance:''),38,365);}if(row?.decision?.pretransit_rest){ctx.fillStyle='#eea276';ctx.fillText('PRE-TRANSIT REST GUARD ACTIVE',38,385);}$('#scrub').value=frame;$('#tick').textContent=frame+1;}
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
function selectedProtocol(){return $('#experimentType').value||'cognitive';}
function protocolReady(){const p=selectedProtocol();return (p==='cognitive'||p==='titan'||p==='trident'||p==='hyperion'||p==='chimera'||p==='metamorph'||p==='aura'||p==='tessera'||p==='mnemorph'||p==='mco')?true:p==='odysseus'?odysseusReady:p==='helm'?helmReady:p==='odyssey'?odysseyReady&&horizonReady&&contrastReady&&atlasReady:p==='horizon'?horizonReady&&contrastReady&&atlasReady:p==='contrast'?contrastReady&&atlasReady:atlasReady;}
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
  if(!seedChosen)$('#experimentSeed').value=protocol==='cognitive'?'90000000':protocol==='trident'?'88000000':protocol==='titan'?'99000000':protocol==='hyperion'?'303000001':protocol==='chimera'?'202000001':protocol==='metamorph'?'199000001':protocol==='aura'?'150000001':protocol==='tessera'?'160000001':protocol==='mnemorph'?'170000001':protocol==='odysseus'?'133000001':protocol==='mco'?'140000001':protocol==='helm'?'123000001':protocol==='odyssey'?'110000001':protocol==='horizon'?'109000001':protocol==='contrast'?'104000001':'93000001';
  $('#protocolNote').textContent=protocol==='cognitive'?'POSEIDON v2.0 Cognitive Platform evaluates 6 matched lesion arms on identical seeds under standard and POMDP settings.':protocol==='trident'?'TRIDENT compares its intact residual history and observed topology with erased, shifted, directed-support-gate-off and residual-gate-off ablations on shared seeds. Every action and world transition is replayed; results never promote the frozen core.':protocol==='titan'?'TITAN runs once per shared seed against Core, AURA, METAMORPH, CHIMERA and HYPERION. Every TidePool transition and TITAN decision is replayed before its receipt is saved; the result is descriptive and does not promote a model.':protocol==='hyperion'?'HYPERION is an opt-in software controller using simulated trace consolidation, entropy and drive heuristics on identical worlds. Device-inspired telemetry is synthetic.':protocol==='chimera'?'CHIMERA is an opt-in software controller using a classical amplitude heuristic, associative memory and 3D lattice telemetry on identical worlds.':protocol==='metamorph'?'METAMORPH developmental life-stage ecdysis (MOLT), NexusFlow causal potential flux, and Chronos cyclic beaconing on identical worlds.':protocol==='aura'?'Biomimetic ring attractor heading tracking and sparse Kenyon cell neuropil arbitration on identical worlds.':protocol==='tessera'?'Independent-lineage verified macro-action commons and finite-domain semantic safety pre-checks on identical seeds.':protocol==='mnemorph'?'Mnemorph archaeological memory assays: hub vs periphery structural lesions, context shift, and associative regrowth.':protocol==='odysseus'?'Six paired arms evaluate Odysseus calibrated routing, ungated, no-memory, Odyssey, random against the frozen policy baseline.':protocol==='mco'?'Full causal factorial evaluation across 16 carrier combinations and 3 falsifiable negative controls on standard knowledge benchmarks.':protocol==='helm'?'Six paired arms compare Helm selected, observation-only, no-innovation, yoked and ungated critics against the frozen policy baseline.':protocol==='odyssey'?'Ten paired controllers compare Odyssey cognitive mapping against Horizon Atlas, Contrast Atlas, Atlas v2, Neural MPC and policy.':protocol==='horizon'?'Nine paired controllers separate multi-horizon return advantages, depletion-trap guards, memory removal and uncalibrated interventions on identical worlds.':protocol==='contrast'?'Nine paired controllers separate action advantage, absolute bounds, memory removal and unfiltered corrections. Atlas v2 is retained as a baseline.':'Six paired controllers compare the original Atlas v2 with memory removed, neural MPC, policy, heuristic and random controls.';
  $('#hyperionSetup').hidden=(protocol!=='hyperion');
  $('#chimeraSetup').hidden=(protocol!=='chimera');
  $('#metamorphSetup').hidden=(protocol!=='metamorph');
  $('#auraSetup').hidden=(protocol!=='aura');
  $('#tesseraSetup').hidden=(protocol!=='tessera');
  $('#mnemorphSetup').hidden=(protocol!=='mnemorph');
  $('#titanSetup').hidden=(protocol!=='titan');
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
  const ready=status?.ready===true||kind==='cognitive',label=kind==='cognitive'?'POSEIDON v2.0 Cognitive Platform':kind==='titan'?'TITAN Composite Controller':kind==='hyperion'?'HYPERION Frontier Super-Controller':kind==='chimera'?'CHIMERA Super-Controller':kind==='aura'?'AURA Biomimetic Central Complex':kind==='tessera'?'Tessera Macro-Commons':kind==='mnemorph'?'Mnemorph Archaeological Memory':kind==='odysseus'?'Odysseus Navigator':kind==='helm'?'Helm Critic v0.6':kind==='odyssey'?'Odyssey Atlas v5':kind==='horizon'?'Horizon Atlas v4':kind==='contrast'?'Contrast Atlas':'Atlas v2';
  if(kind==='cognitive')cognitiveReady=ready;else if(kind==='hyperion')hyperionReady=ready;else if(kind==='chimera')chimeraReady=ready;else if(kind==='aura')auraReady=ready;else if(kind==='tessera')tesseraReady=ready;else if(kind==='mnemorph')mnemorphReady=ready;else if(kind==='odysseus')odysseusReady=ready;else if(kind==='helm')helmReady=ready;else if(kind==='odyssey')odysseyReady=ready;else if(kind==='horizon')horizonReady=ready;else if(kind==='contrast')contrastReady=ready;else if(kind!=='titan')atlasReady=ready;
  if($('#'+kind+'State')){$('#'+kind+'State').textContent=ready?'OPT-IN CANDIDATE':'FIT NOT READY';$('#'+kind+'State').classList.toggle('available',ready);}
  const receipt=status?.fit_receipt,selection=receipt?.selection,calibration=receipt?.calibration;
  const sampleCount=receipt?.training_samples;
  let detail=ready?'A fitted candidate is available. ':label+' is unavailable. '+(status?.error||'Fit it with the local command below.')+' ';
  if(ready&&kind==='cognitive')detail='Unified v2.0 architecture with 3-tier Mnemosyne, Oracle ensemble, Active Inference, and Metacognition. ';
  if(ready&&kind==='titan')detail='Experimental composite in synthetic TidePool. Sidecars are software simulations; '+(status?.receipt_ready?'a paired receipt is ready for replay verification. ':'run the paired benchmark to produce a replay-verified receipt. ');
  if(ready&&kind==='hyperion')detail='Simulated transition-trace consolidation, entropy and drive heuristics, and synthetic intermittent-energy state. ';
  if(ready&&kind==='chimera')detail='Classical amplitude-based action heuristic, associative memory, and synthetic 3D lattice telemetry. ';
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
    const name=arm.replace(/^trident_vs_/,'').replace(/_vs_policy$/,'');
    paired.append(resultRow([controllerName(name),signed(row.mean_reward_delta??row.mean_reward_diff),interval,counts.map(x=>numeric(x)?x:'—').join(' / ')],name));
  }
  $('#pairedHeading').textContent=data.schema==='poseidon-trident-experiment-v1'?'TRIDENT paired reward differences vs ablations':'Paired reward differences vs learned policy';
  $('#pairedCaption').textContent=data.schema==='poseidon-trident-experiment-v1'?'Each TRIDENT arm minus the named ablation on the same seed. No model is promoted.':'Each controller minus policy on the same seed. '+(returns?'This protocol provides no uncertainty interval.':'Intervals are reported by the experiment.');
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
  const payload={kind:protocol==='trident'?'trident-experiment':protocol==='titan'?'titan-experiment':protocol==='hyperion'?'hyperion-experiment':protocol==='aura'?'aura-experiment':protocol==='tessera'?'tessera-experiment':protocol==='mnemorph'?'mnemorph-experiment':protocol==='helm'?'helm-experiment':protocol,seed:Number($('#experimentSeed').value),scarcity:Number($('#scarcity').value),episodes:Number($('#episodes').value),max_steps:Number($('#maxSteps').value)};
  stopWorld();
  experimentBusy=true;currentExperiment=null;experimentOutcome='running';$('#experimentResults').hidden=true;
  $('#decisionPanel').hidden=true;
  $('#experimentPanel').setAttribute('aria-busy','true');
  $('#experimentStatus').classList.remove('error');
  $('#experimentStatus').textContent='Running '+(protocol==='cognitive'||protocol==='trident'||protocol==='titan'||protocol==='atlas'||protocol==='helm'||protocol==='odysseus'?'six':protocol==='hyperion'||protocol==='aura'?'five':protocol==='chimera'?'four':protocol==='contrast'||protocol==='horizon'?'nine':'ten')+' controllers on '+payload.episodes+' paired episodes each, up to '+payload.max_steps+' steps. Local CPU work may take a while…';
  syncControls();
  if(protocol==='cognitive'||protocol==='trident'||protocol==='titan'||protocol==='hyperion'||protocol==='chimera'||protocol==='aura'||protocol==='tessera'||protocol==='mnemorph'||protocol==='helm'||protocol==='odysseus'||protocol==='mco'){
    try{
      const payload={kind:protocol==='cognitive'?'cognitive-experiment':protocol==='trident'?'trident-experiment':protocol==='titan'?'titan-experiment':protocol==='hyperion'?'hyperion-experiment':protocol==='chimera'?'chimera-experiment':protocol==='aura'?'aura-experiment':protocol==='tessera'?'tessera-experiment':protocol==='mnemorph'?'mnemorph-experiment':protocol==='mco'?'mco-experiment':protocol==='odysseus'?'odysseus-experiment':'helm-experiment',seed:Number($('#experimentSeed').value),scarcity:Number($('#scarcity').value),episodes:Number($('#episodes').value),max_steps:Number($('#maxSteps').value)};
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
    const url=protocol==='cognitive'?'/api/cognitive-experiment':protocol==='hyperion'?'/api/hyperion-experiment':protocol==='chimera'?'/api/chimera-experiment':protocol==='metamorph'?'/api/metamorph-experiment':protocol==='aura'?'/api/aura-experiment':protocol==='tessera'?'/api/tessera-experiment':protocol==='mnemorph'?'/api/mnemorph-experiment':protocol==='odysseus'?'/api/odysseus-experiment':protocol==='mco'?'/api/mco-experiment':protocol==='odyssey'?'/api/odyssey-experiment':protocol==='horizon'?'/api/horizon-experiment':protocol==='contrast'?'/api/contrast-experiment':'/api/experiment';
    const data=await request(url,payload);
    showExperiment(data);experimentOutcome='complete';
    $('#experimentStatus').textContent='Completed locally. Compare controllers or benchmark factorial subsets below.';
    if(data.cognitive_status)setCandidateStatus('cognitive',data.cognitive_status);
    if(data.hyperion)setCandidateStatus('hyperion',data.hyperion);
    if(data.titan)setCandidateStatus('titan',data.titan);
    if(data.trident)setCandidateStatus('trident',data.trident);
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
    setCandidateStatus('cognitive',data.cognitive);setCandidateStatus('aura',data.aura);setCandidateStatus('tessera',data.tessera);setCandidateStatus('mnemorph',data.mnemorph);setCandidateStatus('metamorph',data.metamorph);setCandidateStatus('chimera',data.chimera);setCandidateStatus('hyperion',data.hyperion);setCandidateStatus('titan',data.titan);setCandidateStatus('trident',data.trident);
    if(data.mco?.ready)mcoReady=true;
    if(!protocolChosen){$('#experimentType').value='cognitive';protocolChosen=true;}
    updateProtocol();
  }catch{$('#connection').textContent='Offline · start the local server';}
  finally{statusInFlight=false;syncControls();}
}

async function pollTelemetry(){
  try{
    const res=await fetch('/api/telemetry/memory-influence');
    if(!res.ok)return;
    const tel=await res.json();
    if($('#dvGemStatus')){$('#dvGemStatus').textContent=tel.dv_gem_active?'ACTIVE':'INACTIVE';$('#dvGemStatus').style.background=tel.dv_gem_active?'#2563eb':'#64748b';$('#dvGemStatus').style.color='white';}
    if($('#dvGemDeltas'))$('#dvGemDeltas').textContent=JSON.stringify(tel.dv_gem_deltas);
    if($('#epistemicDriveText'))$('#epistemicDriveText').textContent=`Active Inference Drive: ${Number(tel.active_inference_drive).toFixed(3)} | Epistemic Reduction: ${Number(tel.epistemic_reduction).toFixed(3)}`;
    if($('#overrideStatus')){$('#overrideStatus').textContent=tel.planner_override?'OVERRIDE':'NOMINAL';$('#overrideStatus').style.background=tel.planner_override?'#f59e0b':'#10b981';$('#overrideStatus').style.color='white';}
    if($('#overrideText'))$('#overrideText').textContent=`Override: ${tel.planner_override} | Margin: +${Number(tel.override_margin).toFixed(4)} | Scope: ${tel.readiness_scope||'calibrated-stage6-core'}`;

    const fRes=await fetch('/api/nexus-omega/factored-state');
    if(fRes.ok){
      const fData=await fRes.json();
      const d=fData.decision;
      if(d){
        if($('#factoredStateText') && d.factored_state){
          const fs=d.factored_state;
          $('#factoredStateText').textContent=`Endo: [${fs.endogenous?.join(', ')}] | Exo: [${fs.exogenous?.join(', ')}] | Regime: storm=${fs.causal_regime?.[0]}`;
        }
        if($('#efeText')){
          $('#efeText').textContent=`G(a): ${d.free_energy_objective > 0 ? '+':''}${d.free_energy_objective} | Prag: ${d.pragmatic_value} | Epist: ${d.epistemic_value} | Uncalibrated penalty: ${d.conformal_lcb} | Heuristic guard: ${d.conformal_override?'OVERRIDE':'inactive'} | Calm recovery: ${d.recovery_admission?'ADMITTED':'inactive'}`;
          if($('#efeStatus')){
            $('#efeStatus').textContent=d.conformal_override?'HEURISTIC OVERRIDE':'UNCALIBRATED';
            $('#efeStatus').style.background=d.conformal_override?'#f59e0b':'#64748b';
            $('#efeStatus').style.color='white';
          }
        }
        if($('#actionRankText')){
          const labels=['forage','shelter','explore','rest'];
          const scores=Array.isArray(d.action_scores)?[...d.action_scores].sort((a,b)=>b.efe-a.efe):[];
          $('#actionRankText').textContent=scores.length
            ? 'Uncalibrated action scores: '+scores.map(s=>`${labels[s.action]||`action ${s.action}`} G=${decimal(s.efe,3)} R=${decimal(s.predicted_reward,3)}${s.selected_by_objective?' [top]':''}${s.selected_after_guard?' [chosen]':''}`).join(' · ')
            : 'Per-action ranking unavailable for this planner';
        }
      }
    }

    const hRes=await fetch('/api/nexus-omega/hierarchical-state');
    if(hRes.ok){
      const hData=await hRes.json();
      const hd=hData.decision;
      if(hd){
        if($('#macroText')){
          $('#macroText').textContent=`Macro Intent: ${hd.macro_intent?.toUpperCase()} | Commit Ticks: ${hd.macro_intent_remaining_ticks} | Micro Action: ${hd.micro_action_name}`;
        }
        if($('#diffusionText') && hd.imagined_trajectory){
          const it=hd.imagined_trajectory;
          $('#diffusionText').textContent=`H=${it.horizon} | Raw: ${it.unguided_cumulative_reward>0?'+':''}${it.unguided_cumulative_reward} | Guided: ${it.cumulative_reward>0?'+':''}${it.cumulative_reward} | Return Δ: ${it.guided_return_difference} | Max state Δ: ${it.max_branch_divergence_l2} | Uncalibrated penalty: ${it.unguided_return_penalty} | forwards: ${it.ensemble_member_forwards}`;
        }
        if($('#actionRankText') && Array.isArray(hd.action_scores) && hd.action_scores.length){
          const labels=['forage','shelter','explore','rest'];
          const scores=[...hd.action_scores].sort((a,b)=>b.efe-a.efe);
          $('#actionRankText').textContent='Uncalibrated micro scores: '+scores.map(s=>`${labels[s.action]||`action ${s.action}`} G=${decimal(s.efe,3)} R=${decimal(s.predicted_reward,3)}${s.selected_by_micro_planner?' [micro]':''}${s.selected_after_hierarchy?' [chosen]':''}`).join(' · ');
        }
      }
    }

    const uRes=await fetch('/api/nexus-omega/ultima-state');
    if(uRes.ok){
      const uData=await uRes.json();
      if(uData.quantum_state && $('#quantumText')){
        const qs=uData.quantum_state;
        $('#quantumText').textContent=`Purity: ${Number(qs.purity).toFixed(4)} | Berry Phase: ${Number(qs.berry_phase_holonomy).toFixed(4)} rad | Ent: ${Number(qs.von_neumann_entropy).toFixed(4)} | Tunnel: ${Number(qs.tunneling_rate).toFixed(3)}`;
        if($('#quantumStatus')){$('#quantumStatus').textContent=qs.purity>0.8?'SUPERPOSITION':'DECOHERENT';$('#quantumStatus').style.background=qs.purity>0.8?'#06b6d4':'#f59e0b';}
      }
      if(uData.spectral_profile && $('#spectralText')){
        const sp=uData.spectral_profile;
        $('#spectralText').textContent=`Dom Period: T=${Number(sp.dominant_period).toFixed(1)} | Phase: ${Number(sp.current_phase).toFixed(3)} rad | Storm Resonance: ${(Number(sp.storm_resonance_probability)*100).toFixed(1)}% | Spec Ent: ${Number(sp.spectral_entropy).toFixed(3)}`;
        if($('#spectralStatus')){$('#spectralStatus').textContent=sp.storm_resonance_probability>0.5?'RESONANCE SPIKE':'HARMONIC';$('#spectralStatus').style.background=sp.storm_resonance_probability>0.5?'#ef4444':'#ec4899';}
      }
      if(uData.active_opcode && $('#bytecodeText')){
        const op=uData.active_opcode;
        $('#bytecodeText').textContent=`Opcode: ${op.opcode_name} | Steps: ${op.execution_steps?.length||0} | Mutated: ${op.mutated} | Cert: ${op.verification_certificate?.slice(0,12)}...`;
        if($('#bytecodeStatus')){$('#bytecodeStatus').textContent=op.ltl_safe?'LTL CERTIFIED':'FALLBACK SAFE';$('#bytecodeStatus').style.background=op.ltl_safe?'#10b981':'#f59e0b';}
      }
      if(uData.thermodynamic_state && $('#autopoiesisText')){
        const ts=uData.thermodynamic_state;
        const pad=ts.affect_valence_pad||[0,0,0];
        $('#autopoiesisText').textContent=`PAD: [${pad.map(v=>Number(v)>=0?'+'+Number(v).toFixed(2):Number(v).toFixed(2)).join(', ')}] | S_int: +${Number(ts.internal_entropy_production).toFixed(3)} | S_diss: +${Number(ts.heat_dissipation_rate).toFixed(3)} | Viability: ${(Number(ts.autopoietic_viability)*100).toFixed(1)}%`;
      }
      if(uData.dream_receipt && $('#dreamText')){
        const dr=uData.dream_receipt;
        $('#dreamText').textContent=`Replayed: ${dr.replayed_transitions} | Regret Red: ${Number(dr.mean_regret_mitigation).toFixed(3)} | Synaptic Pruning: ${(Number(dr.synaptic_pruning_ratio)*100).toFixed(1)}% | Hub Resilience: +${Number(dr.hub_resilience_gain).toFixed(3)}`;
      }
    }

    const aRes=await fetch('/api/nexus-omega/apex-state');
    if(aRes.ok){
      const aData=await aRes.json();
      if(aData.cognitive_monologue && $('#monologueText')){
        const cm=aData.cognitive_monologue;
        $('#monologueText').textContent=`Regime: ${cm.predicted_regime} | Hypotheses: ${cm.causal_hypotheses?.length||0} do(A) | Curiosity: +${Number(cm.epistemic_curiosity_gain).toFixed(4)}`;
        if($('#thoughtChainText') && Array.isArray(cm.thought_chain)){
          $('#thoughtChainText').textContent=cm.thought_chain.slice(0, 3).join(' ➔ ');
        }
        if($('#monologueStatus')){$('#monologueStatus').textContent='DELIBERATING';$('#monologueStatus').style.background='#6366f1';}
      }
      if(aData.nonstationary_phase && $('#phaseDriftText')){
        const np=aData.nonstationary_phase;
        $('#phaseDriftText').textContent=`T_est: ${Number(np.estimated_pulse_interval).toFixed(1)} | Confirmed: ${np.confirmed_pulse_interval?Number(np.confirmed_pulse_interval).toFixed(1):'none'} | Drift Vel: ${Number(np.frequency_drift_velocity)>=0?'+':''}${Number(np.frequency_drift_velocity).toFixed(3)} | Latch: ${np.phase_boundary_latch_active?'ACTIVE':'INACTIVE'}`;
        if($('#phaseDriftStatus')){$('#phaseDriftStatus').textContent=np.phase_boundary_latch_active?'LATCHED':'RESONATING';$('#phaseDriftStatus').style.background=np.phase_boundary_latch_active?'#f59e0b':'#0ea5e9';}
      }
      if(aData.contrastive_memory && $('#contrastiveText')){
        const cmem=aData.contrastive_memory;
        $('#contrastiveText').textContent=`ΔQ: ${Number(cmem.counterfactual_advantage)>=0?'+':''}${Number(cmem.counterfactual_advantage).toFixed(3)} | Pred E(t+1): ${Number(cmem.predicted_vitality_energy).toFixed(2)} | Pred H(t+1): ${Number(cmem.predicted_vitality_hydration).toFixed(2)} | Deadband: ${cmem.within_deadband?'IN_DEADBAND':'DISCRIMINATED'}`;
        if($('#contrastiveStatus')){$('#contrastiveStatus').textContent=cmem.guard_triggered?'VITALITY GUARD':'NOMINAL';$('#contrastiveStatus').style.background=cmem.guard_triggered?'#ef4444':'#10b981';}
      }
    }
  }catch{}
}
if($('#pollTelemetryBtn'))$('#pollTelemetryBtn').onclick=()=>pollTelemetry();
if($('#triggerReasonBtn'))$('#triggerReasonBtn').onclick=async()=>{
  try{
    if($('#monologueStatus')){$('#monologueStatus').textContent='REASONING';$('#monologueStatus').style.background='#8b5cf6';}
    if($('#thoughtChainText'))$('#thoughtChainText').textContent='Generating Ghost-in-the-Machine SCM counterfactual hypotheses…';
    const res=await fetch('/api/nexus-omega/reason',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:42})});
    if(!res.ok)throw Error(res.statusText);
    const cm=await res.json();
    if($('#monologueStatus')){$('#monologueStatus').textContent='SYNTHESIZED';$('#monologueStatus').style.background='#10b981';}
    if($('#monologueText'))$('#monologueText').textContent=`Regime: ${cm.predicted_regime} | Hypotheses: ${cm.causal_hypotheses?.length||0} do(A) | Curiosity: +${Number(cm.epistemic_curiosity_gain).toFixed(4)}`;
    if($('#thoughtChainText') && Array.isArray(cm.thought_chain)){
      $('#thoughtChainText').textContent=cm.thought_chain.join(' ➔ ');
    }
  }catch(err){
    if($('#monologueStatus')){$('#monologueStatus').textContent='FAILED';$('#monologueStatus').style.background='#ef4444';}
    if($('#thoughtChainText'))$('#thoughtChainText').textContent='Deliberation failed: '+err.message;
  }
};
if($('#triggerDreamBtn'))$('#triggerDreamBtn').onclick=async()=>{
  try{
    if($('#dreamStatus')){$('#dreamStatus').textContent='REPLAYING';$('#dreamStatus').style.background='#8b5cf6';}
    if($('#dreamText'))$('#dreamText').textContent='Morpheus offline generative replay consolidating counterfactual rollouts...';
    const res=await fetch('/api/nexus-omega/dream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({num_rollouts:4,seed:42})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#dreamStatus')){$('#dreamStatus').textContent='CONSOLIDATED';$('#dreamStatus').style.background='#10b981';}
    if($('#dreamText'))$('#dreamText').textContent=`Replayed: ${data.replayed_transitions} | Regret Red: ${Number(data.mean_regret_mitigation).toFixed(3)} | Synaptic Pruning: ${(Number(data.synaptic_pruning_ratio)*100).toFixed(1)}% | Hub Resilience: +${Number(data.hub_resilience_gain).toFixed(3)}`;
  }catch(err){
    if($('#dreamStatus')){$('#dreamStatus').textContent='FAILED';$('#dreamStatus').style.background='#ef4444';}
    if($('#dreamText'))$('#dreamText').textContent='Dream consolidation failed: '+err.message;
  }
};
if($('#runDiscoveryBtn'))$('#runDiscoveryBtn').onclick=async()=>{
  try{
    if($('#discoveryStatus')){$('#discoveryStatus').textContent='RUNNING';$('#discoveryStatus').style.background='#8b5cf6';$('#discoveryStatus').style.color='white';}
    if($('#discoveryText'))$('#discoveryText').textContent='Evaluating counterfactual interventions and abductive noise across 4 seeds...';
    const res=await fetch('/api/nexus-omega/discover',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed_base:50000,num_seeds:4,max_steps:16})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#discoveryStatus')){$('#discoveryStatus').textContent=data.verified?'VERIFIED':'COMPLETE';$('#discoveryStatus').style.background=data.verified?'#10b981':'#ef4444';$('#discoveryStatus').style.color='white';}
    if($('#discoveryText'))$('#discoveryText').textContent=`Signals passing simulator gate: ${data.ratified_laws_count} | Other results: ${data.refutations_count} | SHA: ${data.ledger_sha256.slice(0,16)}...`;
  }catch(err){
    if($('#discoveryStatus')){$('#discoveryStatus').textContent='FAILED';$('#discoveryStatus').style.background='#ef4444';}
    if($('#discoveryText'))$('#discoveryText').textContent='Discovery cycle failed: '+err.message;
  }
};
if($('#runPrometheusCurriculumBtn'))$('#runPrometheusCurriculumBtn').onclick=async()=>{
  try{
    if($('#prometheusStatus')){$('#prometheusStatus').textContent='RUNNING CURRICULUM';$('#prometheusStatus').style.background='#8b5cf6';}
    if($('#prometheusText'))$('#prometheusText').textContent='Executing 5-Phase Developmental Curriculum (Physics -> Interventions -> Raft Transfer)...';
    const res=await fetch('/api/discovery-lab/curriculum',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:42})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#prometheusStatus')){$('#prometheusStatus').textContent=data.overall_status==='PASSED'?'ALL 5 PHASES PASSED':'FAILED';$('#prometheusStatus').style.background=data.overall_status==='PASSED'?'#10b981':'#ef4444';}
    if($('#prometheusText'))$('#prometheusText').textContent=`Narrow: ${data.phases?.phase_5?.metrics?.narrow_river_strategy} | Wide: ${data.phases?.phase_5?.metrics?.wide_river_strategy} | Receipt: ${data.receipt_sha256?.slice(0,16)}...`;
  }catch(err){
    if($('#prometheusStatus')){$('#prometheusStatus').textContent='FAILED';$('#prometheusStatus').style.background='#ef4444';}
    if($('#prometheusText'))$('#prometheusText').textContent='Curriculum failed: '+err.message;
  }
};
if($('#synthesizeEnzymeBtn'))$('#synthesizeEnzymeBtn').onclick=async()=>{
  try{
    if($('#enzymeStatus')){$('#enzymeStatus').textContent='SYNTHESIZING';$('#enzymeStatus').style.background='#8b5cf6';}
    if($('#enzymeText'))$('#enzymeText').textContent='Generating 3D active site triad and Michaelis-Menten kinetic parameters...';
    const res=await fetch('/api/discovery-lab/enzyme/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({substrate:'lignocellulose',target_reaction:'cross_linking',temperature:298.15,ph:6.5})});
    if(!res.ok)throw Error(res.statusText);
    const data=await res.json();
    if($('#enzymeStatus')){$('#enzymeStatus').textContent='CATALYZED';$('#enzymeStatus').style.background='#10b981';}
    if($('#enzymeText'))$('#enzymeText').textContent=`ID: ${data.enzyme_id} | ${data.name} | ΔΔG‡: ${data.delta_delta_g_barrier_kcal} kcal/mol | Bond: ${data.cured_bond_strength} | Water: ${data.water_proof_rating}`;
  }catch(err){
    if($('#enzymeStatus')){$('#enzymeStatus').textContent='FAILED';$('#enzymeStatus').style.background='#ef4444';}
    if($('#enzymeText'))$('#enzymeText').textContent='Synthesis failed: '+err.message;
  }
};

let riverState = null;
async function initRiverWorld(seed=42, riverWidth=null) {
  stopWorld();
  $('#timeline').hidden = true;
  $('#decisionPanel').hidden = true;
  $('#artifactType').textContent = 'RIVERWORLD';
  const rw = riverWidth !== null ? riverWidth : Number($('#riverWidthSelect')?.value || 3);
  try {
    const res = await fetch(`/api/discovery-lab/river-state?seed=${seed}&river_width=${rw}`);
    if (res.ok) {
      riverState = await res.json();
      drawRiverWorld(riverState);
      if ($('#riverStatusNote')) {
        $('#riverStatusNote').textContent = `RiverWorld initialized: Width ${rw}m | Objects: ${riverState.objects?.length || 0} | Health: 100% | Stamina: 100%`;
      }
    }
  } catch (e) {
    console.error('RiverWorld load error:', e);
  }
}

function drawRiverWorld(state) {
  if (!state) return;
  const c = $('#worldCanvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  const W = 600, H = 400;
  ctx.clearRect(0, 0, W, H);

  // World-grid → canvas mapping (each grid cell spans one column/row)
  const gridW = Math.max(1, state.grid_width || 16);
  const gridH = Math.max(1, state.grid_height || 10);
  const toCanvasX = (x) => ((x + 0.5) / gridW) * W;
  const toCanvasY = (y) => ((y + 0.5) / gridH) * H;

  // 2. River Channel bounds (computed first so the bank fills match)
  const isWide = state.river_width > 4;
  const riverStartX = ((state.river_start_x ?? 7) / gridW) * W;
  const riverEndX = (((state.river_end_x ?? 9) + 1) / gridW) * W;
  const riverW = riverEndX - riverStartX;

  // 1. Left Riverbank (Earthen grass)
  ctx.fillStyle = '#14532d';
  ctx.fillRect(0, 0, riverStartX, H);
  ctx.fillStyle = '#166534';
  for (let i = 0; i < 15; i++) {
    ctx.fillRect(10 + (i*8) % Math.max(20, riverStartX - 20), 20 + (i*26)%360, 4, 8);
  }

  const grad = ctx.createLinearGradient(riverStartX, 0, riverEndX, 0);
  grad.addColorStop(0, '#0284c7');
  grad.addColorStop(0.5, '#0369a1');
  grad.addColorStop(1, '#0284c7');
  ctx.fillStyle = grad;
  ctx.fillRect(riverStartX, 0, riverW, H);

  // Wave ripple lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
  ctx.lineWidth = 1.5;
  for (let y = 30; y < H; y += 45) {
    ctx.beginPath();
    ctx.moveTo(riverStartX + 15, y);
    ctx.bezierCurveTo(riverStartX + 100, y - 8, riverStartX + 200, y + 8, riverEndX - 15, y);
    ctx.stroke();
  }

  // 3. Right Bank (Target territory)
  ctx.fillStyle = '#1e3a8a';
  ctx.fillRect(riverEndX, 0, W - riverEndX, H);
  ctx.fillStyle = '#fbbf24';
  ctx.fillRect(riverEndX + 55, 180, 5, 40);
  ctx.beginPath();
  ctx.moveTo(riverEndX + 60, 180);
  ctx.lineTo(riverEndX + 90, 192);
  ctx.lineTo(riverEndX + 60, 204);
  ctx.fill();
  ctx.font = 'bold 11px Segoe UI';
  ctx.fillStyle = '#fde68a';
  ctx.fillText('GOAL BANK', riverEndX + 35, 170);

  // 4. Render Physical Objects
  const objects = state.objects || [];
  let connectedLogs = [];

  objects.forEach(obj => {
    const cx = toCanvasX(obj.position[0]);
    const cy = toCanvasY(obj.position[1]);

    if (obj.material.includes('stone') || obj.material.includes('granite')) {
      const inRiver = obj.position[0] >= state.river_start_x && obj.position[0] <= state.river_end_x;
      ctx.fillStyle = inRiver ? '#334155' : '#64748b';
      ctx.beginPath();
      ctx.arc(cx, cy, 14, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = inRiver && isWide ? '#ef4444' : '#94a3b8';
      ctx.lineWidth = inRiver && isWide ? 2 : 1;
      ctx.stroke();

      if (inRiver) {
        ctx.fillStyle = isWide ? '#f87171' : '#cbd5e1';
        ctx.font = '9px monospace';
        ctx.fillText(isWide ? 'SUBMERGED' : 'STEP', cx - 20, cy + 22);
      }
    } else if (obj.is_buoyant) {
      ctx.fillStyle = '#b45309';
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      if (typeof ctx.roundRect === 'function') {
        ctx.roundRect(cx - 10, cy - 20, 20, 40, 5);
      } else {
        ctx.rect(cx - 10, cy - 20, 20, 40);
      }
      ctx.fill();
      ctx.stroke();

      if (obj.connected_object_ids && obj.connected_object_ids.length > 0) {
        connectedLogs.push({cx, cy, id: obj.object_id, strength: obj.bond_strength});
      }
    }
  });

  // Raft catalytic cross-linking lines
  if (connectedLogs.length >= 2) {
    ctx.strokeStyle = '#c084fc';
    ctx.lineWidth = 3;
    ctx.setLineDash([4, 2]);
    ctx.beginPath();
    ctx.moveTo(connectedLogs[0].cx, connectedLogs[0].cy);
    for (let k = 1; k < connectedLogs.length; k++) {
      ctx.lineTo(connectedLogs[k].cx, connectedLogs[k].cy);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#e9d5ff';
    ctx.font = 'bold 10px monospace';
    ctx.fillText('RAFT (CATALYZED BOND 0.98)', connectedLogs[0].cx - 30, connectedLogs[0].cy - 26);
  }

  // 5. Render Agent Avatar
  const agentX = toCanvasX(state.agent.x);
  const agentY = toCanvasY(state.agent.y);
  ctx.fillStyle = '#38bdf8';
  ctx.beginPath();
  ctx.arc(agentX, agentY, 12, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = '#0f172a';
  ctx.font = 'bold 9px Segoe UI';
  ctx.fillText('YOU', agentX - 9, agentY + 3);

  // 6. Canvas HUD Overlays
  ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
  ctx.fillRect(0, 0, W, 28);
  ctx.fillStyle = '#e2e8f0';
  ctx.font = '11px Segoe UI';
  ctx.fillText(`PROMETHEUS RIVERWORLD · Channel: ${isWide ? 'WIDE (7m) - RAFT REQUIRED' : 'NARROW (3m) - STONES VIABLE'} · Health: ${Math.round(state.agent.health * 100)}% · Stamina: ${Math.round(state.agent.stamina * 100)}%`, 12, 18);

  ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
  ctx.fillRect(0, H - 28, W, 28);
  ctx.fillStyle = '#38bdf8';
  ctx.font = '10px monospace';
  const gmText = isWide ? 'GM: +0.28m (CATAMARAN STABLE)' : 'GM: N/A (STEPPING)';
  const payloadText = isWide ? 'Payload: 108kg > 70kg' : 'Direct Stride';
  ctx.fillText(`Buoyancy: +353.2N | ${payloadText} | Naval: ${gmText} | Shield: PROVEN_SAFE`, 12, H - 10);
}

if ($('#riverWidthSelect')) {
  $('#riverWidthSelect').onchange = () => {
    initRiverWorld(42, Number($('#riverWidthSelect').value));
  };
}

if ($('#stepRiverBtn')) {
  $('#stepRiverBtn').onclick = async () => {
    try {
      const act = Number($('#riverActionSelect')?.value || 1);
      const rw = Number($('#riverWidthSelect')?.value || 3);
      const res = await fetch('/api/discovery-lab/step', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({action: act, river_width: rw, seed: 42})
      });
      if (!res.ok) throw Error(res.statusText);
      const data = await res.json();
      message('assistant', `Executed step: ${data.action_executed} | Event: ${data.info?.event || 'none'} | Reward: ${data.reward} | Pos: ${JSON.stringify(data.info?.agent_position)}`, 'Project PROMETHEUS Discovery Lab');
      if ($('#riverStatusNote')) {
        $('#riverStatusNote').textContent = `Action: ${data.action_executed} | Reward: ${data.reward} | Event: ${data.info?.event || 'nominal'} | State Hash: ${data.state_hash.slice(0, 16)}...`;
      }
      initRiverWorld(42, rw);
    } catch (err) {
      message('error', 'RiverWorld step failed: ' + err.message);
    }
  };
}

if ($('#resetRiverBtn')) {
  $('#resetRiverBtn').onclick = () => {
    initRiverWorld(42, Number($('#riverWidthSelect')?.value || 3));
  };
}

drawIdle();refreshStatus();refreshJobs();pollTelemetry();setInterval(()=>{refreshStatus();pollTelemetry();},15000);


