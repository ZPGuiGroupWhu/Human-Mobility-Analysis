export const initArg = {
  // select 页面
  curId: -1, // box-chart 实例唯一标识
  allData: null, // chart 图表数据源
  selectedUsers: [], // 筛选得到的用户编号数组(交集)
  selectedByCharts: [], // 图表筛选结果
  selectedByCalendar: [], // 日历筛选结果

  // predict 页面
  selectedTraj: {}, // 选择的单轨迹对象
}