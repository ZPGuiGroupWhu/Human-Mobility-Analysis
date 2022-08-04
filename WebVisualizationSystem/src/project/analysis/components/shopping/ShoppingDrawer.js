import React, { useEffect, useLayoutEffect, useState } from 'react';
import '../Content.scss';
import './ShoppingDrawer.scss';
import SingleCard from './SingleCard';
import { useSelector, useDispatch } from 'react-redux';
import { Button, Tooltip } from 'antd';
import { DeleteOutlined, CloseCircleOutlined, PlusSquareOutlined, MinusSquareOutlined } from '@ant-design/icons';
import { delSelectTraj, clearSelectTraj } from '@/app/slice/analysisSlice';

export default function ShoppingDrawer(props) {
  const {
    ShenZhen, // ShenZhen.json
    // 抽屉大小
    drawerWidth = '170px',
    drawerHeight = '360px',
    drawerPadding = '10px',
  } = props;

  const imageWidth = (parseInt(drawerWidth) - 2 * parseInt(drawerPadding) - 2) + 'px'; // 图片大小

  const selectTrajs = useSelector(state => state.analysis.selectTrajs);
  const dispatch = useDispatch();


  // 存储多选框结果 - trajectory id
  const [checks, setChecks] = useState([]);  // 标记数组
  const [glbChecked, setGlbChecked] = useState(false);  // 全选

  // 标记数组为空时，重置全选功能
  // useEffect(() => {
  //   if (!checks.length) {
  //     setGlbChecked(false)
  //   }
  // }, [checks.length])


  useEffect(()=>{
    if(checks.length===0 || checks.length !== selectTrajs.length){
      // 非全选
      setGlbChecked(false);
    }else{
      // 全选
      setGlbChecked(true);
    }
  }, [checks.length])

  // 候选列表发生变动，自动定位到末尾
  useLayoutEffect(() => {
    let actCtn = document.querySelector('.shopping-drawer-main .single-card-ctn-active');
    if (actCtn) {
      actCtn.scrollIntoView()
    } else {
      let ctn = document.querySelector('.shopping-drawer-main');
      ctn.scrollTop = ctn.scrollHeight;
    }
  }, [selectTrajs])

  return (
    // <SingleCard> 组件可以将一个 Canvas 转为 Image 展示
    // 但是，若 selectTrajs 一开始传入就超过了 Canvas 绘图个数的限制，那么遍历数组时会先绘制所有的 Canvas，然后再转为 Image，此时会报错。
    // 我们期望绘制一个 Canvas 后就立即转为 Image，即选择轨迹时就”同步“进行渲染，因此当前 <ShoppingCart> 组件应该在未选中时全局隐藏，而不是删除 DOM
    <aside
      className={`common-box-style shopping-drawer-ctn`}
      style={{
        padding: drawerPadding,
        // 限制width会出问题，例如当购物车内容过多时，会出现纵向滚动条，滚动条有固定宽度。但是width固定，会导致横向滚动条出现
        minWidth: drawerWidth,
        maxHeight: drawerHeight,
        display: selectTrajs.length ? '' : 'none'
      }}
    >
      <header className='shopping-drawer-header'>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <h1 style={{ display: 'inline-block' }}>{`候选(${checks.length}/${selectTrajs.length})`}</h1>
          <span>
            <Tooltip title="删除">
              <Button
                icon={<DeleteOutlined />}
                disabled={!checks.length}
                size='small'
                onClick={() => {
                  dispatch(delSelectTraj(checks));
                  setChecks([]);
                }}
              ></Button>
            </Tooltip>
            <Tooltip title="清空">
              <Button
                icon={<CloseCircleOutlined />}
                size='small'
                onClick={() => {
                  dispatch(clearSelectTraj());
                  setChecks([]);
                }}
              ></Button>
            </Tooltip>
            {
              !glbChecked ?
                (
                  <Tooltip title="全选">
                    <Button
                      icon={<PlusSquareOutlined />}
                      size='small'
                      onClick={() => {
                        setChecks(selectTrajs.map(item => item.id))
                        setGlbChecked(true);
                      }}
                    ></Button>
                  </Tooltip>
                ) :
                (
                  <Tooltip title="取消">
                    <Button
                      icon={<MinusSquareOutlined />}
                      size='small'
                      onClick={() => {
                        setChecks([]);
                        setGlbChecked(false);
                      }}
                    ></Button>
                  </Tooltip>
                )
            }
          </span>
        </div>
      </header>
      <div className='shopping-drawer-main'>
        {/* 首先加载canvas，在其渲染完成后，调用onAfterRender回调，存储为image，并替换canvas */}
        {selectTrajs.length ?
          selectTrajs.map((item) => {
            return (
              <SingleCard
                key={item.id}
                data={item}
                width={imageWidth}
                ShenZhen={ShenZhen}
                setChecks={setChecks}
                glbChecked={glbChecked}
                checksNumber={checks.length}
              />
            )
          })
          : null}
      </div>
    </aside>
  )
}
