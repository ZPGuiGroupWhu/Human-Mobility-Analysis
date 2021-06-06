// 组件导入
import React from 'react';
import { BrowserRouter as Router, Redirect, Route, Switch } from 'react-router-dom'
import Layout from '@/components/layout/Layout';
import FunctionBar from '@/project/FunctionBar';
import PageAnalysis from '@/project/PageAnalysis';
import PageSelect from '@/project/PageSelect';
import PagePredict from '@/project/PagePredict';
// 自定义 Hook 导入
import { useResize } from '@/common/hooks/useResize';


// Context 对象创建
export const windowResize = React.createContext(false);

function Main() {
  // 窗口 resize 监听
  const isResize = useResize(200);
  const initParams = {
    initCenter: '深圳市',
    initZoom: 12,
  }

  return (
    <windowResize.Provider value={isResize}>
      <Router>
        <Layout
          src='https://picsum.photos/170'
          title='中文xxxxxxxxxxx'
          imgHeight='80%'
        >
          {<FunctionBar />}
          {
            <Switch>
              <Route exact path='/select'>
                <PageSelect {...initParams} />
              </Route>
              <Route exact path='/select/analysis' render={() => <PageAnalysis {...initParams} />} />
              <Route exact path='/select/predict' render={() => <PagePredict {...initParams} />} />
              {/* 若均未匹配，重定向至首页 */}
              <Redirect to='/select' />
            </Switch>
          }
        </Layout>
      </Router>
    </windowResize.Provider>
  );
}

export default Main;