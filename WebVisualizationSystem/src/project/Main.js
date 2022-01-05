// 组件导入
import React, { useState } from 'react';
import { BrowserRouter as Router, Redirect, Route, Switch } from 'react-router-dom';
import BreadCrumb from '@/components/bread-crumb/BreadCrumb.js';
import Layout from '@/components/layout/Layout';
import PageAnalysis from '@/project/analysis/PageAnalysis';
import PageSelect from '@/project/select/PageSelect';
import Predict from '@/project/predict/Predict';
// 自定义 Hook 导入
import { useResize } from '@/common/hooks/useResize';
// 图片
import { earth } from '@/icon';

const windowResize = React.createContext(false);

function Main() {
  // 窗口 resize 监听
  const isResize = useResize(200);
  const initParams = {
    initCenter: '深圳市',
    initZoom: 12,
  }

  const [routes, setRoutes] = useState([
    { breadCrumbName: '用户筛选', targetURL: '/select', status: true },
    { breadCrumbName: '轨迹筛选', targetURL: '/select/analysis', status: false },
    { breadCrumbName: '目的地预测', targetURL: '/select/predict', status: false },
  ])

  return (
    <windowResize.Provider value={isResize}>
      <Router>
        <Layout
          src={earth}
          title='用户移动模式分析平台'
          imgHeight='70%'
        >
          {
            <BreadCrumb
              routes={routes}
            ></BreadCrumb>
          }
          {
            <Switch>
              <Route exact path='/select'>
                <PageSelect {...initParams} setRoutes={setRoutes} />
              </Route>
              <Route exact path='/select/analysis' render={() => <PageAnalysis {...initParams} setRoutes={setRoutes} />} />
              <Route path='/select/predict' component={Predict} />
              {/* 若均未匹配，重定向至首页 */}
              <Redirect to='/select' />
            </Switch>
          }
        </Layout>
      </Router>
    </windowResize.Provider>
  );
}

export {
  windowResize,
}

export default Main;