import React from 'react';
import '../Content.scss';
import './ShoppingCart.scss';
import SingleCard from './SingleCard';

export default function ShoppingCart(props) {
  const {
    selectTrajs,
    ShenZhen,
    isSelected,  // 当前组件是否被选中展示，若不选中则隐藏但不删除 DOM，见 style 设置
    handleDeleteSelectTraj,
  } = props;

  return (
    // <SingleCard> 组件可以将一个 Canvas 转为 Image 展示
    // 但是，若 selectTrajs 一开始传入就超过了 Canvas 绘图个数的限制，那么遍历数组时会先绘制所有的 Canvas，然后再转为 Image，此时会报错。
    // 我们期望绘制一个 Canvas 后就立即转为 Image，即选择轨迹时就”同步“进行渲染，因此当前 <ShoppingCart> 组件应该在未选中时全局隐藏，而不是删除 DOM
    <div className={`analysis-common-line-ctn ${isSelected ? 'shopping-cart-show' : 'shopping-cart-hidden'}`}>
      <div className='shopping-cart-ctn'>
        {/* 首先加载canvas，在其渲染完成后，调用onAfterRender回调，存储为image，并替换canvas */}
        {selectTrajs.length ?
          selectTrajs.map((item) => {
            // deckgl 数据组织
            const OD = [{
              O: item.data[0],
              D: item.data.slice(-1)[0],
              sourceColor: [252, 252, 46],
              targetColor: [255, 77, 41],
            }];
            const path = [{
              path: item.data,
              color: [254, 137, 20],
            }];
            return (
              <SingleCard
                key={item.id}
                id={item.id}
                OD={OD}
                path={path}
                ShenZhen={ShenZhen}
                handleDeleteSelectTraj={handleDeleteSelectTraj}
              />
            )
          })
          : null}
      </div>
    </div>
  )
}
