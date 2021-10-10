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
import { logo } from '@/icon';

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
    { breadCrumbName: '轨迹筛选', targetURL: '/select/analysis', status: true },
    { breadCrumbName: '目的地预测', targetURL: '/select/predict', status: false },
  ])

  // setRoutes(prev => {
  //   const newRoutes = _.cloneDeep(prev);
  //   newRoutes[1].status = true;
  //   return newRoutes;
  // })

  return (
    <windowResize.Provider value={isResize}>
      <Router>
        <Layout
          src={logo}
          title='轨迹目的地预测平台'
          imgHeight='60%'
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
              <Route path='/select/predict' render={() => <Predict {...initParams} />} />
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