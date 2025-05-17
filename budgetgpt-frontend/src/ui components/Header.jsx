import React from 'react'
import { DialogBox } from './DialogBox'
function Header () {
  return (
    <div>
      <header>
        <nav className='flex justify-between p-4 bg-zinc-950 align-middle items-center'>
          <div>
            <h1 className='font-semibold text-white text-2xl'>BudgetGPT</h1>
          </div>
          <DialogBox />
        </nav>
      </header>
    </div>
  )
}

export default Header
