document.addEventListener('alpine:init', () => {
  Alpine.data('studio', () => ({
    ...window.analysisControls(),
    ...window.modelingControls(),
    ...window.deepControls(),
    dataset: null, loading: true, submitting: false, error: '', notice: '', job: null, tab: 'data',
    preview: {rows: [], page: 1, total: 0, pages: 1}, previewLoading: false, previewSequence: 0,
    page: 1, pageSize: 50, search: '', sort: '', direction: 'asc', filters: {}, showFilters: false,
    filterColumn: '', filterValue: '', selectedColumns: [], selectedRows: [], cell: null, range: null,
    rangeStart: 1, rangeEnd: 10, editValue: '', editNull: false, detection: null,
    importOpen: false, exportOpen: false, clearOpen: false, exportFormat: 'csv', downloadUrl: '',
    maxUpload: 200, importHosts: [], file: null, sourceUrl: '', sourceChoices: null, uploadId: '', chosenSources: [],
    parseOptions: {delimiter: '', encoding: '', header: true, regex_delimiter: false, depth: 3, record_key: '', xpath: './*'},
    action: 'fill', method: 'mean', value: '', group: '', target: '', neighbors: 5, threshold: 50,
    outlierThreshold: 1.5, dateFormat: '', pattern: '', replacement: '', order: '', newName: '', cleanDelimiter: ',',
    histogramVisible: false, histogramPosition: '', pollTimer: null,
    actions: [['fill','Fill missing values'],['drop_selected_columns','Drop selected columns'],['drop_missing','Drop rows with missing values'],['drop_columns','Drop mostly empty columns'],
      ['detect_duplicates','Detect duplicate rows'],['drop_duplicates','Remove duplicate rows'],['detect_outliers','Detect outliers'],
      ['convert','Convert data type'],['text','Clean text'],['encode','Encode categories'],['scale','Scale / normalize'],
      ['rename','Rename a column'],['split','Split a column'],['merge','Combine columns']],
    methodMap: {
      fill: [['mean','Mean'],['median','Median'],['mode','Mode'],['constant','Constant value'],['ffill','Forward fill'],['bfill','Backward fill'],['interpolate','Linear interpolation'],['knn','KNN imputation'],['group_mode','Most frequent per group']],
      drop_missing: [['any','Any missing value'],['all','All selected values missing']],
      detect_outliers: [['iqr','Interquartile range (IQR)'],['zscore','Z-score'],['isolation_forest','Isolation forest']],
      convert: [['numeric','Numeric'],['string','String'],['datetime','Datetime'],['category','Category']],
      text: [['trim','Trim whitespace'],['lower','Lowercase'],['upper','Uppercase'],['title','Title case'],['regex','Regex find & replace'],['special','Strip special characters']],
      encode: [['onehot','One-hot encoding'],['label','Label encoding'],['ordinal','Ordinal encoding'],['target','Target encoding']],
      scale: [['standard','Standard scaler'],['minmax','Min-max scaler'],['robust','Robust scaler'],['log','Log transform (log1p)']],
    },
    get columns() { return this.dataset?.profile.columns || []; },
    get busy() { return this.submitting || ['queued','running'].includes(this.job?.status); },
    get methods() { return this.methodMap[this.action] || []; },
    get selectionLabel() {
      if (this.cell) return `1 cell · ${this.cell.column} · row ${this.cell.row+1}`;
      let cols = this.selectedColumns.length ? `${this.selectedColumns.length} columns` : 'All columns';
      return cols + (this.range ? ` · rows ${this.range[0]+1}–${this.range[1]+1}` : this.selectedRows.length ? ` · ${this.selectedRows.length} rows` : ' · all rows');
    },
    fmt(n) { return Number(n || 0).toLocaleString(); },
    missingPercent() { return (100*this.dataset.profile.missing/Math.max(1,this.dataset.profile.rows*this.dataset.profile.column_count)).toFixed(1); },
    async api(path, options={}) {
      const csrf = document.querySelector('[name=csrfmiddlewaretoken]').value;
      const response = await fetch('/api/'+path, {credentials:'same-origin', ...options,
        headers: {'X-CSRFToken':csrf, ...(options.body instanceof FormData ? {} : {'Content-Type':'application/json'}), ...options.headers}});
      let data; try { data = await response.json(); } catch { throw new Error('The server could not respond. Check the connection and try again.'); }
      if (!response.ok) throw new Error(typeof data.detail==='string' ? data.detail : JSON.stringify(data));
      return data;
    },
    async initStudio() {
      this.$watch('method', value => { if(this.action==='detect_outliers') this.outlierThreshold = value==='iqr'?1.5:value==='zscore'?3:.05; });
      this.$watch('tab', value => this.$nextTick(()=>{if(['statistics','models','deep'].includes(value))this.drawAnalysisCharts();if(value==='deep')this.drawDeepLive();if(value==='data')this.drawHeatmap();}));
      await this.refresh(true);
    },
    async refresh(resume=false) {
      try {
        const state = await this.api('state/'); this.dataset=state.dataset; this.maxUpload=state.max_upload_mb; this.importHosts=state.import_hosts;
        this.syncAnalysis(state);
        this.syncModels();
        this.syncDeep(state);
        if (resume) {
          const active=state.jobs.find(j=>['queued','running'].includes(j.status));
          if (active) { this.job=active; this.updateDeepJob();this.schedulePoll(); }
          else if(state.jobs[0]?.status==='failed') {this.error=state.jobs[0].message;this.job=state.jobs[0];this.updateDeepJob();}
          else if(state.jobs[0]?.result?.choose) this.showChoices(state.jobs[0].result);
          const download=state.jobs.find(j=>j.kind==='export' && j.result.download_url); if(download) this.downloadUrl=download.result.download_url;
        }
        this.loading=false;
        if(this.dataset) { await this.loadPreview(); this.$nextTick(()=>this.drawHeatmap()); }
      } catch(e) { this.error=e.message; this.loading=false; }
    },
    async loadPreview() {
      if(!this.dataset) return;
      const seq=++this.previewSequence; this.previewLoading=true;
      const query = new URLSearchParams({page:this.page, page_size:this.pageSize, q:this.search, filters:JSON.stringify(this.filters), direction:this.direction});
      if(this.sort) query.set('sort',this.sort); if(this.detection) query.set('detection',this.detection.id);
      try { const data=await this.api('preview/?'+query); if(seq===this.previewSequence) { this.preview=data; this.page=data.page; } }
      catch(e) { if(seq===this.previewSequence) this.error=e.message; }
      finally { if(seq===this.previewSequence) this.previewLoading=false; }
    },
    resetView() { this.clearSelection(); this.filters={};this.search='';this.sort='';this.page=1;this.detection=null; },
    clearSelection() { this.selectedColumns=[];this.selectedRows=[];this.cell=null;this.range=null; },
    toggleColumn(c) { this.cell=null;this.selectedColumns=this.selectedColumns.includes(c)?this.selectedColumns.filter(x=>x!==c):[...this.selectedColumns,c]; },
    selectAllColumns(on) { this.cell=null;this.selectedColumns=on?this.columns.map(c=>c.name):[]; },
    toggleRow(row) { this.cell=null;this.range=null;this.selectedRows=this.selectedRows.includes(row)?this.selectedRows.filter(x=>x!==row):[...this.selectedRows,row]; },
    selectCell(row,column,event) {
      if(event.shiftKey && this.cell) {
        const a=this.columns.findIndex(c=>c.name===this.cell.column),b=this.columns.findIndex(c=>c.name===column);
        this.selectedColumns=this.columns.slice(Math.min(a,b),Math.max(a,b)+1).map(c=>c.name);
        this.range=[Math.min(this.cell.row,row.id),Math.max(this.cell.row,row.id)];this.selectedRows=[];this.cell=null;return;
      }
      this.selectedColumns=[column];this.selectedRows=[row.id];this.range=null;
      this.cell={row:row.id,column,truncated:row.truncated.includes(column)};
      this.editValue=row.values[column]===null?'':String(row.values[column]);this.editNull=row.values[column]===null;
      this.$nextTick(()=>document.getElementById('inline-cell-editor')?.focus());
    },
    isSelected(row,column) { return (!this.selectedColumns.length || this.selectedColumns.includes(column)) &&
      (this.range ? row>=this.range[0] && row<=this.range[1] : this.selectedRows.length ? this.selectedRows.includes(row) : this.selectedColumns.length>0); },
    setRange() { if(!Number.isInteger(this.rangeStart)||!Number.isInteger(this.rangeEnd)||this.rangeStart<1||this.rangeEnd<this.rangeStart||this.rangeEnd>this.dataset.profile.rows) {this.error='Enter a valid row range within the dataset.';return;} this.cell=null;this.selectedRows=[];this.range=[this.rangeStart-1,this.rangeEnd-1]; },
    scope() { const selection={columns:[...this.selectedColumns]};if(this.range)selection.range=[...this.range];else if(this.selectedRows.length)selection.rows=[...this.selectedRows];return selection; },
    sortBy(column) { this.direction=this.sort===column && this.direction==='asc'?'desc':'asc';this.sort=column;this.page=1;this.loadPreview(); },
    setFilter() { if(!this.filterColumn)return;this.filters={...this.filters,[this.filterColumn]:this.filterValue};this.page=1;this.loadPreview(); },
    resetMethod() { this.method=this.methods[0]?.[0] || ''; },
    async apply() {
      if(this.action==='drop_selected_columns' && !this.selectedColumns.length){this.error='Select the columns you want to drop.';return;}
      const operation={action:this.action, selection:this.scope()}; if(this.method)operation.method=this.method;
      if(this.action==='fill') { if(this.method==='constant')operation.value=this.value;if(this.method==='group_mode')operation.group=this.group;if(this.method==='knn')operation.neighbors=this.neighbors; }
      if(this.action==='drop_columns')operation.threshold=this.threshold;
      if(this.action==='detect_outliers')operation[this.method==='isolation_forest'?'contamination':'threshold']=this.outlierThreshold;
      if(this.action==='convert' && this.method==='datetime')operation.format=this.dateFormat;
      if(this.action==='text' && this.method==='regex') {operation.pattern=this.pattern;operation.replacement=this.replacement;}
      if(this.action==='encode' && this.method==='ordinal') {try{operation.order=JSON.parse(this.order);}catch{this.error='Enter the ordinal order as a JSON list.';return;}}
      if(this.action==='encode' && this.method==='target')operation.target=this.target;
      if(['rename','merge'].includes(this.action))operation.name=this.newName;
      if(['split','merge'].includes(this.action))operation.delimiter=this.cleanDelimiter;
      await this.submitJob(this.action.startsWith('detect')?'detect':'clean',{operation});
    },
    async editCell() { if(this.cell && !this.cell.truncated) await this.submitJob('clean',{operation:{action:'edit',selection:{columns:[this.cell.column],rows:[this.cell.row]},value:this.editNull?null:this.editValue}}); },
    async removeOutliers() { await this.submitJob('clean',{operation:{action:'remove_outliers'},detection_job:this.detection.id}); },
    async submitJob(kind,extra={}) {
      if(this.busy || !this.dataset)return;this.submitting=true;this.error='';this.notice='';
      try {const job=await this.api('jobs/',{method:'POST',body:JSON.stringify({kind,dataset_id:this.dataset.id,revision:this.dataset.revision,revision_id:this.dataset.revision_id,...extra})});await this.acceptJob(job);}
      catch(e){this.error=e.message;}finally{this.submitting=false;}
    },
    async acceptJob(job) { this.job=job;this.updateDeepJob();if(['queued','running'].includes(job.status))this.schedulePoll();else await this.finishJob(); },
    schedulePoll() { clearTimeout(this.pollTimer);this.pollTimer=setTimeout(()=>this.poll(),1000); },
    async poll() {
      try {this.job=await this.api('jobs/'+this.job.id+'/');this.updateDeepJob();if(['queued','running'].includes(this.job.status))this.schedulePoll();else await this.finishJob();}
      catch(e){this.error=e.message;clearTimeout(this.pollTimer);this.pollTimer=setTimeout(()=>this.poll(),5000);}
    },
    async finishJob() {
      if(this.job.status==='failed'){this.error=this.job.message;return;}
      const result=this.job.result;
      if(result.choose) {this.showChoices(result);return;}
      if(this.job.kind==='image_ingest'){const id=this.job.id;await this.refresh();this.deepOptions.image_id=id;this.tab='deep';this.notice='Image collection validated. Select CNN settings to train.';return;}
      if(['analysis','model','predict','deep'].includes(this.job.kind)){const id=this.job.id;await this.refresh();await this.loadReport(id);this.notice='Complete. Your report and outputs are saved.';return;}
      if(this.job.kind==='detect'){this.detection=this.job;await this.loadPreview();return;}
      if(this.job.kind==='export'){this.downloadUrl=result.download_url;this.notice='Your export is ready. Open Export data to download it again.';const a=document.createElement('a');a.href=this.downloadUrl;a.download='';a.click();return;}
      this.resetView();this.notice='Saved. Your workspace is up to date.';await this.refresh();
    },
    openImport() {this.sourceChoices=null;this.uploadId='';this.chosenSources=[];this.importOpen=true;},
    showChoices(result){this.sourceChoices=result.choose;this.uploadId=result.upload_id;this.chosenSources=[];this.importOpen=true;},
    toggleSource(name){this.chosenSources=this.sourceChoices.kind==='tables'?[name]:this.chosenSources.includes(name)?this.chosenSources.filter(x=>x!==name):[...this.chosenSources,name];},
    async importData() {
      if(this.busy)return;this.error='';this.notice='';
      if(this.file && this.file.size>this.maxUpload*1024*1024){this.error='File exceeds the '+this.maxUpload+' MB limit.';this.importOpen=false;return;}
      if(this.sourceChoices && !this.chosenSources.length){this.error='Choose a worksheet or table.';this.importOpen=false;return;}
      const options={...this.parseOptions};if(!options.record_key)delete options.record_key;
      if(this.sourceChoices?.kind==='sheets')options.sheets=[...this.chosenSources];
      if(this.sourceChoices?.kind==='tables')options.table=this.chosenSources[0];
      const body=new FormData();body.append('options',JSON.stringify(options));
      if(this.uploadId)body.append('upload_id',this.uploadId);else if(this.sourceUrl)body.append('url',this.sourceUrl);else if(this.file)body.append('file',this.file);else return;
      this.submitting=true;this.importOpen=false;
      try {await this.acceptJob(await this.api('ingest/',{method:'POST',body}));}catch(e){this.error=e.message;}finally{this.submitting=false;}
    },
    async loadSample(){try{const response=await fetch('/static/sample.csv');this.file=new File([await response.blob()],'water-quality-sample.csv',{type:'text/csv'});this.sourceUrl='';this.uploadId='';this.sourceChoices=null;await this.importData();}catch(e){this.error=e.message;}},
    async replay(event){const file=event.target.files[0];if(!file)return;if(file.size>1024*1024){this.error='Recipes must be under 1 MB.';return;}try{const recipe=JSON.parse(await file.text());await this.submitJob('replay',{recipe});}catch(e){this.error='Invalid recipe: '+e.message;}event.target.value='';},
    operationLabel(op){return op.action==='import'?'Original import':(this.actions.find(([a])=>a===op.action)?.[1] || op.action.replaceAll('_',' '))+(op.method?' · '+op.method.replaceAll('_',' '):'');},
    drawHeatmap(){const target=document.getElementById('heatmap');if(!target || !window.Plotly || !this.dataset)return;Plotly.react(target,[{type:'heatmap',z:this.dataset.profile.heatmap,x:this.columns.map(c=>c.name),colorscale:[[0,'#eaf4ee'],[1,'#e8a938']],zmin:0,zmax:1,showscale:false,hovertemplate:'%{x}<br>Row group %{y}<br>Missing: %{z:.0%}<extra></extra>'}],{margin:{l:0,r:0,t:3,b:0},paper_bgcolor:'transparent',plot_bgcolor:'transparent',xaxis:{visible:false},yaxis:{visible:false},height:140},{displayModeBar:false,responsive:true});},
    hoverHistogram(column,event){if(!column.histogram || !window.Plotly)return;this.histogramPosition=`left:${Math.min(event.clientX,window.innerWidth-270)}px;top:${Math.min(event.clientY+20,window.innerHeight-185)}px`;this.histogramVisible=true;this.$nextTick(()=>Plotly.react(this.$refs.histogram,[{type:'bar',x:column.histogram.x,y:column.histogram.y,marker:{color:'#237a60'}}],{title:{text:column.name,font:{size:12}},margin:{l:30,r:10,t:35,b:25},height:174,width:254,font:{size:9},paper_bgcolor:'#fff'},{displayModeBar:false,staticPlot:true}));},
    hideHistogram(){this.histogramVisible=false;},
    destroy(){clearTimeout(this.pollTimer);},
  }));
});
