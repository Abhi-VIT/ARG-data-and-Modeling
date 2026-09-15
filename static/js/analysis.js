// Mixed into the existing Alpine workspace; reports remain scoped to saved revisions.
window.analysisControls = () => ({
  analysisType: 'descriptive', analysisColumns: [], recentReports: [], report: null,
  reportId: '', reportLoading: false, reportSequence: 0, reportTablePages: {}, chartExport:null,
  analysisOptions: {method:'auto', model:'multiple', target:'', stepwise:'none', criterion:'aic',
    confidence:.95, vif_threshold:5, lag:10, p_enter:.05, p_exit:.1,
    anovaMethod:'oneway', group:'', group2:'', subject:'', tukey:false, tukey_factor:'', alpha:.05,
    plot:'histogram', x:'', y:'', bins:30},
  interactionA:'', interactionB:'', interactionTerms:[], polynomialColumns:[], polynomialDegree:2,
  analysisTypes:[['descriptive','Descriptive statistics'],['correlation','Correlation & association'],
    ['regression','Regression diagnostics'],['anova','Analysis of variance'],['eda','Exploratory plots']],
  analysisNumeric(column) {return /^(u?int|float|decimal|bool)/i.test(column.dtype);},
  numericAnalysis() {
    return this.analysisType==='regression' ||
      (this.analysisType==='correlation' && ['pearson','spearman','kendall'].includes(this.analysisOptions.method)) ||
      (this.analysisType==='eda' && ['histogram','box','violin','scatter','pair','heatmap'].includes(this.analysisOptions.plot));
  },
  analysisSelected() {return this.columns.filter(c=>this.analysisColumns.includes(c.name));},
  analysisPredictors() {return this.analysisSelected().filter(c=>this.analysisNumeric(c) && c.name!==this.analysisOptions.target);},
  excludedAnalysisColumns() {return this.numericAnalysis()?this.analysisSelected().filter(c=>!this.analysisNumeric(c)).map(c=>c.name):[];},
  selectAnalysisAll() {this.analysisColumns=this.columns.filter(c=>!this.numericAnalysis() || this.analysisNumeric(c)).map(c=>c.name);},
  toggleAnalysisColumn(name) {this.analysisColumns=this.analysisColumns.includes(name)?this.analysisColumns.filter(c=>c!==name):[...this.analysisColumns,name];},
  syncAnalysis(state) {
    this.recentReports=[...(state.analyses || []),...(state.models || [])].sort((a,b)=>b.created_at.localeCompare(a.created_at));
    const valid=this.columns.map(c=>c.name);
    this.analysisColumns=this.analysisColumns.filter(c=>valid.includes(c));
    this.polynomialColumns=this.polynomialColumns.filter(c=>valid.includes(c));
    this.interactionTerms=this.interactionTerms.filter(pair=>pair.every(c=>valid.includes(c)));
    for(const field of ['target','group','group2','subject','tukey_factor','x','y'])
      if(!valid.includes(this.analysisOptions[field]))this.analysisOptions[field]='';
  },
  showStatistics() {
    this.tab='statistics';
    if(this.report && ['ml','prediction'].includes(this.report.analysis_type)){this.report=null;this.reportId='';}
    if(!this.analysisColumns.length && this.selectedColumns.length)this.analysisColumns=[...this.selectedColumns];
    if(!this.report) {
      const latest=this.recentReports.find(j=>j.kind==='analysis' && j.result.dataset_id===this.dataset?.id);
      if(latest)this.loadReport(latest.id);
    } else this.$nextTick(()=>this.drawAnalysisCharts());
  },
  visibleReports(){return this.recentReports.filter(j=>this.tab==='models'?['model','predict'].includes(j.kind):j.kind==='analysis');},
  correlationSuggestion() {
    const columns=this.analysisSelected();
    if(columns.length!==2)return 'Auto chooses a method for each pair using observed types and levels. You can override it below.';
    const [a,b]=columns;
    if(a.unique===2 && b.unique===2)return 'Suggested: Phi coefficient — both columns have two levels.';
    if((a.unique===2 && this.analysisNumeric(b)) || (b.unique===2 && this.analysisNumeric(a)))return 'Suggested: point-biserial — a binary column paired with a numeric column.';
    if(this.analysisNumeric(a) && this.analysisNumeric(b))return 'Suggested: Pearson for linear association; use Spearman or Kendall for rank association.';
    if(!this.analysisNumeric(a) && !this.analysisNumeric(b))return 'Suggested: Cramér’s V — both columns are categorical.';
    return 'This pair needs a different encoding or a manually chosen compatible method.';
  },
  addInteraction() {
    const valid=this.analysisPredictors().map(c=>c.name);
    if(!valid.includes(this.interactionA)||!valid.includes(this.interactionB)||this.interactionA===this.interactionB){this.error='Choose two different selected predictors.';return;}
    const pair=[this.interactionA,this.interactionB].sort();
    if(!this.interactionTerms.some(p=>JSON.stringify(p)===JSON.stringify(pair)))this.interactionTerms.push(pair);
  },
  togglePolynomial(name) {this.polynomialColumns=this.polynomialColumns.includes(name)?this.polynomialColumns.filter(c=>c!==name):[...this.polynomialColumns,name];},
  formulaPreview() {
    const predictors=this.analysisPredictors().map(c=>c.name);
    let terms=[...predictors];
    if(this.analysisOptions.model==='custom') {
      for(const c of this.polynomialColumns.filter(c=>predictors.includes(c)))
        for(let power=2;power<=this.polynomialDegree;power++)terms.push(c+'^'+power);
      terms.push(...this.interactionTerms.map(pair=>pair.join(' × ')));
    }
    return (this.analysisOptions.target || 'Outcome')+' ~ '+(terms.join(' + ') || 'select predictors');
  },
  async runAnalysis() {
    if(!this.analysisColumns.length){this.error='Select columns before running an analysis.';return;}
    const analysis={type:this.analysisType,columns:[...this.analysisColumns]};
    const options=this.analysisOptions;
    if(this.analysisType==='correlation')analysis.method=options.method;
    if(this.analysisType==='regression') {
      for(const key of ['model','target','stepwise','criterion','confidence','vif_threshold','lag','p_enter','p_exit'])analysis[key]=options[key];
      if(options.model==='custom') {
        analysis.interactions=this.interactionTerms.map(pair=>[...pair]);
        analysis.polynomials=Object.fromEntries(this.polynomialColumns.map(c=>[c,Number(this.polynomialDegree)]));
      }
    }
    if(this.analysisType==='anova') {
      for(const key of ['target','group','group2','subject','alpha'])analysis[key]=options[key];
      analysis.method=options.anovaMethod;
      analysis.tukey=['oneway','twoway'].includes(options.anovaMethod) && options.tukey;
      if(options.tukey_factor)analysis.tukey_factor=options.tukey_factor;
    }
    if(this.analysisType==='eda')for(const key of ['plot','x','y','bins'])analysis[key]=options[key];
    await this.submitJob('analysis',{analysis});
  },
  async loadReport(id) {
    if(!id)return;
    const seq=++this.reportSequence;this.reportLoading=true;this.error='';
    try {
      const result=await this.api('jobs/'+id+'/report/');
      if(seq!==this.reportSequence)return;
      this.report=result;this.reportId=id;this.reportTablePages={};this.tab=['ml','prediction'].includes(result.analysis_type)?'models':'statistics';this.thresholdIndex=50;
      if(this.tab==='models' && !this.modelCatalog.length)this.loadModelCatalog();
      this.$nextTick(()=>this.drawAnalysisCharts());
    } catch(e){this.error=e.message;}
    finally{if(seq===this.reportSequence)this.reportLoading=false;}
  },
  analysisValue(value) {
    if(value===null || value===undefined)return '—';
    if(typeof value==='boolean')return value?'Yes':'No';
    if(typeof value==='number') {
      if(!Number.isFinite(value))return '—';
      return value!==0 && Math.abs(value)<.0001?value.toExponential(3):value.toLocaleString(undefined,{maximumFractionDigits:6});
    }
    return String(value);
  },
  reportPage(index) {return this.reportTablePages[index] || 1;},
  reportRows(index) {const start=(this.reportPage(index)-1)*30;return this.report.tables[index].rows.slice(start,start+30);},
  changeReportPage(index,delta) {this.reportTablePages={...this.reportTablePages,[index]:Math.max(1,Math.min(Math.ceil(this.report.tables[index].rows.length/30),this.reportPage(index)+delta))};},
  drawAnalysisCharts() {
    if(!this.report || !['statistics','models'].includes(this.tab) || !window.Plotly)return;
    this.report.figures.forEach((plot,index)=>{
      const element=document.getElementById('analysis-chart-'+index);
      if(element) {
        const specification=JSON.parse(JSON.stringify(plot.figure));
        Plotly.react(element,specification.data,specification.layout,{responsive:true,displaylogo:false,
          toImageButtonOptions:{format:'png',filename:'arg-analysis-'+(index+1),scale:2}});
      }
    });
  },
  async downloadChart(index) {
    const element=document.getElementById('analysis-chart-'+index);if(!element)return;
    try {
      const url=await Plotly.toImage(element,{format:'png',scale:2});
      this.chartExport={url,name:'arg-analysis-'+(index+1)+'.png',title:this.report.figures[index].title};
    } catch(e){this.error='Could not prepare the chart image. Try again after the chart finishes loading.';}
  },
});
