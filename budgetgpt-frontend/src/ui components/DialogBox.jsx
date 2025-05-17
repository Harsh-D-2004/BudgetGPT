import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
function CheckIcon (props) {
  return (
    <svg
      {...props}
      xmlns='http://www.w3.org/2000/svg'
      width='24'
      height='24'
      viewBox='0 0 24 24'
      fill='none'
      stroke='currentColor'
      strokeWidth='2'
      strokeLinecap='round'
      strokeLinejoin='round'
    >
      <path d='M20 6 9 17l-5-5' />
    </svg>
  )
}
export function DialogBox () {
  const examplePrompts = [
    'How does the new budget impact job creation in India?',
    'What are the criticisms regarding tax changes for the middle class?',
    'Can you explain the significance of MSME support in the budget?',
    'What are the budget’s implications for India’s rural economy?',
    'How might the budget affect the stock market and investment strategies?'
  ]
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant='outline'>Prompts</Button>
      </DialogTrigger>
      <DialogContent className='max-w-[350px] md:max-w-[550px] '>
        <DialogHeader>
          <DialogTitle>How to Use our Chat app</DialogTitle>
          <DialogDescription>
            This is a AI powered chat app which can answer all your questions
            related to India's 2024 financial budget. Try the prompts given
            below to get started.
          </DialogDescription>
        </DialogHeader>
        <div className='grid gap-2 py-4'>
          {examplePrompts.map((prompt, index) => (
            <div key={index} className='flex items-center gap-2'>
              <CheckIcon className='h-4 w-4 text-green-500' />
              <span>{prompt}</span>
            </div>
          ))}
        </div>

        <DialogFooter className='sm:justify-start'>
          <DialogClose asChild>
            <Button>Close</Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
