"""End-to-end test of the actual generated page. Requires Chrome and websocket-client.

Starts an isolated browser profile and an ephemeral local HTTP server. Does not use
or modify an existing browser session. Optional CHROME_PATH overrides discovery.
"""
import base64
import functools
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import websocket

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'verification'
OUT.mkdir(exist_ok=True)
QUICK = '--quick' in sys.argv
chrome = os.environ.get('CHROME_PATH') or next((p for p in [
    shutil.which('google-chrome'), shutil.which('chromium'),
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
] if p and Path(p).is_file()), None)
if not chrome:
    raise RuntimeError('Set CHROME_PATH to a Chrome or Chromium executable')

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass
    def handle(self):
        try:
            super().handle()
        except ConnectionResetError:
            pass  # Isolated Chrome may close its keepalive socket on shutdown.

server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=str(ROOT)))
threading.Thread(target=server.serve_forever, daemon=True).start()
profile = Path(tempfile.mkdtemp(prefix='nn-learning-lab-test-'))
proc = subprocess.Popen([chrome, '--headless=new', '--remote-debugging-port=0',
    '--remote-allow-origins=*', '--user-data-dir='+str(profile), '--no-first-run',
    '--disable-extensions', '--disable-background-networking', 'about:blank'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
ws = None
counter = 0
errors = []
report = {}
try:
    portfile=profile/'DevToolsActivePort'
    for _ in range(150):
        if portfile.exists():
            break
        time.sleep(.1)
    port=int(portfile.read_text().splitlines()[0])
    tabs=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json'))
    ws=websocket.create_connection(next(t for t in tabs if t['type']=='page')['webSocketDebuggerUrl'],timeout=60)

    def call(method, params=None):
        global counter
        counter+=1
        ws.send(json.dumps({'id':counter,'method':method,'params':params or {}}))
        while True:
            r=json.loads(ws.recv())
            if r.get('method')=='Runtime.exceptionThrown':
                errors.append(r['params'])
            if r.get('id')==counter:
                if 'error' in r:
                    raise RuntimeError(r['error'])
                return r.get('result',{})

    def evaluate(code):
        r=call('Runtime.evaluate',{'expression':code,'returnByValue':True,'awaitPromise':True})
        if 'exceptionDetails' in r:
            raise RuntimeError(r['exceptionDetails'])
        return r.get('result',{}).get('value')

    def wait_until(code, seconds=60):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if evaluate(code):
                return
            time.sleep(.15)
        raise AssertionError('Timed out: '+code)

    def screenshot(name):
        (OUT/name).write_bytes(base64.b64decode(call('Page.captureScreenshot',{'captureBeyondViewport':True})['data']))

    def click(id):
        evaluate(f'document.getElementById({json.dumps(id)}).click()')

    call('Runtime.enable');call('Page.enable')
    call('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1100,'deviceScaleFactor':1,'mobile':False})
    call('Emulation.setEmulatedMedia',{'features':[{'name':'prefers-color-scheme','value':'light'}]})
    call('Page.navigate',{'url':f'http://127.0.0.1:{server.server_port}/index.html'})
    wait_until("!!document.getElementById('rs-lab')?.__test")
    evaluate("window.T=document.getElementById('rs-lab').__test;window.S=()=>T.getState();window.Q=id=>document.getElementById(id);window.V=id=>Q(id).getBoundingClientRect().height>0;")
    report['initial']=evaluate("({widths:S().widths,parameters:S().nets.map(n=>n.parameterCount()),samples:S().points.length,step:S().step,mandatory:['rs-input','rs-plain','rs-res','rs-target-surface','rs-plain-surface','rs-res-surface','rs-loss-chart','rs-gradient-chart'].every(V),optional:['rs-configuration-panel','rs-correction-panel','rs-values-panel','rs-inspector'].every(id=>!V(id))})")
    assert report['initial']['widths']==[10]*37 and report['initial']['parameters']==[4001,4001]
    assert report['initial']['samples']==1024 and report['initial']['step']==0
    assert report['initial']['mandatory'] and report['initial']['optional']
    assert evaluate("![...Q('rs-gradient-chart').querySelectorAll('text')].some(t=>t.textContent==='35')")
    evaluate("window.scrollTo({top:1500,behavior:'instant'})")
    assert evaluate("document.querySelector('.training-toolbar').getBoundingClientRect().top<20")
    evaluate("window.scrollTo({top:0,behavior:'instant'})")
    screenshot('desktop-default.png')
    evaluate('window.before=JSON.stringify(S().nets)')
    for id in ['show-architecture','show-correction','show-values','show-heatmaps','show-slices']:
        click(id)
    assert evaluate('JSON.stringify(S().nets)===before')
    assert evaluate("['rs-loss-chart','rs-gradient-chart','rs-predictions'].every(V)")
    assert evaluate("document.querySelectorAll('[data-colorbar]').length===4 && document.querySelectorAll('[data-color-tick]').length===12")
    evaluate("Q('rs-gradient-source').value='single';Q('rs-gradient-source').dispatchEvent(new Event('change'));Q('rs-preview-offset').value=.2;Q('rs-preview-offset').dispatchEvent(new Event('input'));")
    assert evaluate('JSON.stringify(S().nets)===before')
    assert evaluate("Math.abs(+Q('rs-plain-slice').dataset.previewWeight-(+Q('rs-plain-slice').dataset.currentWeight)-.2)<1e-10")
    evaluate("Q('rs-plain-weights').querySelectorAll('[data-index]')[3].dispatchEvent(new MouseEvent('click',{bubbles:true}))")
    assert evaluate("+Q('rs-weight-row').value===1 && +Q('rs-weight-col').value===1")
    click('rs-example-step')
    assert evaluate('S().step===1&&S().singleSteps===1&&S().samplesSeen===0&&S().nets.every(n=>n.t===1)')
    assert evaluate("Q('rs-run-status').textContent.includes('selected-example update')")
    click('rs-step')
    assert evaluate("S().step===2&&Q('rs-run-status').textContent.includes('Total 2')")
    click('rs-init');click('preset-small')
    for id in ['show-architecture','show-correction','show-values','show-heatmaps','show-slices']:
        click(id)
    report['runs']=[]
    for id,count in ([] if QUICK else [('rs-run',100),('rs-run-500',500),('rs-run-1000',1000)]):
        before=evaluate('S().step');start=time.monotonic();click(id)
        wait_until('!S().running',120)
        after=evaluate('S().step')
        assert after-before==count,(id,before,after,evaluate("Q('rs-run-status').textContent"))
        report['runs'].append({'updates':count,'seconds':round(time.monotonic()-start,2)})
        print(f'Checked {count} additional updates',flush=True)
    click('rs-run-1000');wait_until('S().running');click('rs-pause')
    paused=evaluate('S().step');time.sleep(.25)
    assert evaluate('!S().running') and evaluate('S().step')==paused
    assert evaluate("Q('rs-run-status').textContent.startsWith('Paused')")
    click('rs-run-1000');click('rs-init');time.sleep(.2)
    assert evaluate('S().step===0&&!S().running&&S().history.length===1')
    evaluate('T.loadPreset(36,10)')
    start=time.monotonic();click('rs-step' if QUICK else 'rs-run-500')
    wait_until('!S().running',180)
    assert evaluate('S().step')==(1 if QUICK else 500)
    report['deepRun']={'seconds':round(time.monotonic()-start,2),'last':evaluate('S().history.at(-1)')}
    screenshot('desktop-trained.png')
    for id in ['show-architecture','show-correction','show-values','show-heatmaps','show-slices','show-widths']:
        click(id)
    evaluate("Q('rs-width-number-5').value=6;Q('rs-width-number-5').dispatchEvent(new Event('change'))")
    assert evaluate('S().step===0&&S().widths[5]===6&&S().nets[1].parameterCount()>S().nets[0].parameterCount()')
    evaluate('T.loadPreset(36,10)')
    report['layouts']=[]
    for width in [390,320]:
        call('Emulation.setDeviceMetricsOverride',{'width':width,'height':900,'deviceScaleFactor':1,'mobile':False})
        time.sleep(.2)
        overflow=evaluate('document.documentElement.scrollWidth>innerWidth')
        assert not overflow,('horizontal overflow',width)
        report['layouts'].append({'width':width,'overflow':overflow})
    screenshot('mobile-panels.png')
    call('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1100,'deviceScaleFactor':1,'mobile':False})
    call('Emulation.setEmulatedMedia',{'features':[{'name':'prefers-color-scheme','value':'dark'}]})
    time.sleep(.3);screenshot('desktop-dark-panels.png')
    call('Page.navigate',{'url':f'http://127.0.0.1:{server.server_port}/guide.html'})
    wait_until("document.title.startsWith('Student guide')")
    assert evaluate("document.querySelectorAll('main h2').length>=6")
    report['errors']=errors
    assert not errors,errors
    report['status']='passed'
    print(json.dumps(report,indent=2),flush=True)
finally:
    if ws:
        try:
            ws.close()
        except Exception:
            pass
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill();proc.wait()
    server.shutdown();server.server_close()
    # Only the unique temporary profile created above can be removed.
    if profile.resolve().parent==Path(tempfile.gettempdir()).resolve() and profile.name.startswith('nn-learning-lab-test-'):
        shutil.rmtree(profile,ignore_errors=True)
    (OUT/('browser-quick-report.json' if QUICK else 'browser-report.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
