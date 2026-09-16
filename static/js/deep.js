window.deepControls=()=>({
  deepOptions:{architecture:'mlp',task:'regression',target:'',features:[],image_id:'',activation:'relu',optimizer:'adam',
    learning_rate:.001,batch_size:32,epochs:30,dropout:.1,early_stopping:true,patience:5,validation_size:.2,test_size:.2,
    seed:42,device:'auto',sequence_length:10,recurrent_layers:1,recurrent_units:32,latent_dim:1,anomaly_quantile:.95,order_column:''},
  deepWidths:'64,32',deepChannels:'16,32',imageSets:[],imageArchive:null,imageFiles:[],deepLive:{},
  showDeep(){this.tab='deep';if(this.report?.analysis_type!=='deep'){this.report=null;this.reportId='';const saved=this.recentReports.find(j=>j.kind==='deep');if(saved)this.loadReport(saved.id);}else this.$nextTick(()=>this.drawAnalysisCharts());this.$nextTick(()=>this.drawDeepLive());},
  syncDeep(state){this.imageSets=state.image_sets||[];const names=this.columns.map(c=>c.name);this.deepOptions.features=this.deepOptions.features.filter(c=>names.includes(c));if(!names.includes(this.deepOptions.target))this.deepOptions.target='';if(!names.includes(this.deepOptions.order_column))this.deepOptions.order_column='';},
  deepSequence(){return ['rnn','lstm','gru'].includes(this.deepOptions.architecture);},
  deepSupervised(){return !['autoencoder','cnn'].includes(this.deepOptions.architecture);},
  selectDeepFeatures(){this.deepOptions.features=this.columns.filter(c=>this.analysisNumeric(c)&&(!this.deepSupervised()||c.name!==this.deepOptions.target)).map(c=>c.name);},
  toggleDeepFeature(name){this.deepOptions.features=this.deepOptions.features.includes(name)?this.deepOptions.features.filter(c=>c!==name):[...this.deepOptions.features,name];},
  async uploadImages(){
    if(this.busy)return;const body=new FormData();
    if(this.imageArchive){if(this.imageArchive.size>this.maxUpload*1024*1024){this.error='Image archive exceeds upload limit.';return;}body.append('archive',this.imageArchive);}
    else if(this.imageFiles.length){if(this.imageFiles.reduce((s,f)=>s+f.size,0)>this.maxUpload*1024*1024){this.error='Combined image size exceeds upload limit.';return;}for(const f of this.imageFiles)body.append('images',f);body.append('paths',JSON.stringify(this.imageFiles.map(f=>f.webkitRelativePath||f.name)));}
    else{this.error='Choose an image ZIP or class-organized folder.';return;}
    this.submitting=true;this.error='';this.notice='';
    try{await this.acceptJob(await this.api('deep/images/',{method:'POST',body}));}catch(e){this.error=e.message;}finally{this.submitting=false;}
  },
  async trainDeep(){
    if(this.busy)return;
    const deep={...this.deepOptions,features:[...this.deepOptions.features],hidden_layers:this.deepWidths.split(',').map(v=>Number(v.trim())),conv_channels:this.deepChannels.split(',').map(v=>Number(v.trim()))};
    this.submitting=true;this.error='';this.notice='';this.deepLive={};
    try{await this.acceptJob(await this.api('deep/train/',{method:'POST',body:JSON.stringify({deep,dataset_id:this.dataset?.id,revision_id:this.dataset?.revision_id})}));}
    catch(e){this.error=e.message;}finally{this.submitting=false;}
  },
  updateDeepJob(){if(this.job?.kind==='deep'){this.deepLive=this.job.result || {};this.$nextTick(()=>this.drawDeepLive());}},
  drawDeepLive(){
    const element=document.getElementById('deep-live-chart');if(!element||!window.Plotly||!this.deepLive.history?.length)return;
    const history=this.deepLive.history;const traces=[];
    for(const key of ['train_loss','validation_loss','train_accuracy','validation_accuracy'])if(key in history[0])traces.push({x:history.map(r=>r.epoch),y:history.map(r=>r[key]),name:key.replaceAll('_',' '),mode:'lines+markers',yaxis:key.includes('accuracy')?'y2':'y'});
    Plotly.react(element,traces,{height:300,margin:{l:50,r:45,t:25,b:45},template:'plotly_white',xaxis:{title:'Epoch'},yaxis:{title:'Loss'},yaxis2:{title:'Accuracy',overlaying:'y',side:'right',range:[0,1]},legend:{orientation:'h'},colorway:['#237a60','#d99a35','#7c6bb1','#3c81a0']},{responsive:true,displaylogo:false});
  },
});
