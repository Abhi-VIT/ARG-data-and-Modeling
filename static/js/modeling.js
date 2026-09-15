window.modelingControls=()=>({
  modelCatalog:[],modelTask:'regression',modelKey:'ridge',modelFeatures:[],modelTarget:'',modelParams:{},
  modelOptions:{test_size:.2,stratify:true,seed:42,scaling:'standard'},
  searchMethod:'none',searchFolds:3,searchIterations:10,searchEnabled:[],searchRanges:{},
  thresholdIndex:50,predictionFile:null,predictionParsing:'{}',predictionThreshold:.5,
  async loadModelCatalog(){try{this.modelCatalog=(await this.api('models/catalog/')).models;this.resetModelParams();}catch(e){this.error=e.message;}},
  availableModels(){return this.modelCatalog.filter(m=>m.task===this.modelTask);},
  modelDefinition(){return this.modelCatalog.find(m=>m.key===this.modelKey);},
  modelParamEntries(){return Object.entries(this.modelDefinition()?.params || {});},
  changeModelTask(){this.modelKey=this.availableModels()[0]?.key || '';this.searchMethod='none';this.resetModelParams();},
  resetModelParams(){this.modelParams=Object.fromEntries(this.modelParamEntries().map(([k,s])=>[k,s.default]));this.searchMethod='none';this.searchEnabled=[];this.searchRanges={};},
  toggleModelFeature(name){this.modelFeatures=this.modelFeatures.includes(name)?this.modelFeatures.filter(c=>c!==name):[...this.modelFeatures,name];},
  selectModelFeatures(){this.modelFeatures=this.columns.map(c=>c.name).filter(c=>!this.supervisedModel() || c!==this.modelTarget);},
  supervisedModel(){return ['regression','classification'].includes(this.modelTask);},
  setModelTarget(){this.modelFeatures=this.modelFeatures.filter(c=>c!==this.modelTarget);},
  toggleSearch(name){this.searchEnabled=this.searchEnabled.includes(name)?this.searchEnabled.filter(k=>k!==name):[...this.searchEnabled,name];if(!this.searchRanges[name])this.searchRanges[name]=JSON.stringify([this.modelParams[name]]);},
  syncModels(){const names=this.columns.map(c=>c.name);this.modelFeatures=this.modelFeatures.filter(c=>names.includes(c));if(!names.includes(this.modelTarget))this.modelTarget='';},
  showModels(){
    this.tab='models';if(!this.modelCatalog.length)this.loadModelCatalog();
    if(!this.report || !['ml','prediction'].includes(this.report.analysis_type)){
      this.report=null;this.reportId='';const saved=this.recentReports.find(j=>['model','predict'].includes(j.kind)&&j.result.dataset_id===this.dataset?.id);if(saved)this.loadReport(saved.id);
    } else this.$nextTick(()=>this.drawAnalysisCharts());
  },
  async trainModel(){
    if(!this.modelFeatures.length){this.error='Select feature columns first.';return;}
    const search={method:this.supervisedModel()?this.searchMethod:'none',folds:Number(this.searchFolds),iterations:Number(this.searchIterations),ranges:{}};
    if(search.method!=='none'){
      try{for(const name of this.searchEnabled){const values=JSON.parse(this.searchRanges[name]);if(!Array.isArray(values)||!values.length)throw Error();search.ranges[name]=values;}}catch{this.error='Search ranges must be JSON arrays, such as [0.1, 1, 10].';return;}
      if(!this.searchEnabled.length){this.error='Select at least one hyperparameter to search.';return;}
    }
    await this.submitJob('model',{model:{model:this.modelKey,features:[...this.modelFeatures],target:this.modelTarget,params:{...this.modelParams},...this.modelOptions,search}});
  },
  thresholdMetrics(){return this.report?.thresholds?.[Number(this.thresholdIndex)] || {};},
  async scoreUpload(){
    if(this.busy||!this.report?.can_predict)return;
    if(!this.predictionFile){this.error='Choose a data file to score.';return;}
    if(this.predictionFile.size>this.maxUpload*1024*1024){this.error='Prediction file exceeds the upload size limit.';return;}
    let options;try{options=JSON.parse(this.predictionParsing);if(!options||Array.isArray(options)||typeof options!=='object')throw Error();}catch{this.error='Parsing options must be a JSON object.';return;}
    const body=new FormData();body.append('file',this.predictionFile);body.append('options',JSON.stringify(options));body.append('threshold',String(this.predictionThreshold));
    this.submitting=true;this.error='';this.notice='';
    try{await this.acceptJob(await this.api('models/'+this.report.model_job_id+'/predict/',{method:'POST',body}));}
    catch(e){this.error=e.message;}finally{this.submitting=false;}
  },
});
